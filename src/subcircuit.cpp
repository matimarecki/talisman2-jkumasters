/*------------------------------------------------------------------------*/
/*! \file subcircuit.cpp
    \brief contains functions necessary to select the subcircuit

  Part of TalisMan
  Copyright(C) 2025 TalisMan-Developers
*/
/*------------------------------------------------------------------------*/
#include "subcircuit.hpp"
#include "linear_algebra.hpp"
#include "prove.hpp"
#include "signal_statistics.hpp"
#include "thread_pool.hpp"

#include <algorithm>// for std::find
#include <bitset>
#include <functional>
#include <random>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>

#include <cmsgen/cmsgen.h>

/*------------------------------------------------------------------------*/

using namespace talisman;

unsigned int seed = 42;

namespace talisman {

using thread_pool = std::unique_ptr<BS::thread_pool<BS::none>>;
extern thread_pool pool;

const size_t batch_size = 64;

// Number of row blocks the uniform (weighted) sampling is split into. This is a
// fixed constant rather than the worker count so that the sampled matrix -- and
// hence every result derived from it -- is identical on every machine.
static constexpr int SAMPLE_BLOCKS = 8;

static std::mt19937_64 gen;
static std::uniform_int_distribution<uint64_t> dis;

void
seed_randomness() {
  gen.seed(seed);
}

// Sample a random 64-bit unsigned integer using Mersenne Twister
uint64_t
sample_uint64() {
  return dis(gen);
}
/*------------------------------------------------------------------------*/
void
fix_missing_children(Subcircuit& result) {
  // Fast lookup sets
  std::unordered_set<Gate*> node_set(result.nodes.begin(), result.nodes.end());
  std::unordered_set<Gate*> input_set(result.inputs.begin(),
                                      result.inputs.end());

  auto add_if_missing = [&](Gate* g) {
    if(!g)
      return;
    if(!node_set.count(g) && !input_set.count(g)) {
      result.inputs.push_back(g);
      input_set.insert(g);
      if(verbose > 2) {
        msg("Added missing child %s to inputs", g->get_var_name().c_str());
      }
    }
  };

  for(Gate* n : result.nodes) {
    // --- child 0 ---
    {
      unsigned ch = n->get_ch0();
      Gate* c = gate(ch);
      add_if_missing(c);
    }

    // --- child 1 ---
    {
      unsigned ch = n->get_ch1();
      Gate* c = gate(ch);
      add_if_missing(c);
    }
  }
}

/*------------------------------------------------------------------------*/
// Recursively ensure that all parents of all inputs in a subcircuit are
// included. If a parent is not in nodes/inputs, it is added to nodes, and its
// parents are processed recursively.
void
ensure_parent_closure(Subcircuit& sc, Gate* root) {
  std::unordered_set<Gate*> node_set(sc.nodes.begin(), sc.nodes.end());
  std::unordered_set<Gate*> input_set(sc.inputs.begin(), sc.inputs.end());

  // Combined set of all gates currently in the subcircuit
  std::unordered_set<Gate*> sc_set = node_set;
  sc_set.insert(input_set.begin(), input_set.end());

  bool changed = true;
  // Keep adding parents whose both children are already in the subcircuit
  while(changed) {
    changed = false;
    for(unsigned i = 0; i < num_gates; ++i) {
      Gate* p = gates[i];
      if(!p)
        continue;
      // skip input gates (they are not nodes to add)
      if(p->get_input())
        continue;
      if(p->get_level() > root->get_level())
        continue;

      Gate* c0 = gate(p->get_ch0());
      Gate* c1 = gate(p->get_ch1());
      if(!c0 || !c1)
        continue;

      // if both children are part of the subcircuit (either as input or node)
      if(sc_set.count(c0) && sc_set.count(c1)) {
        if(!node_set.count(p)) {
          // if p was listed as an input, remove it from inputs
          if(input_set.count(p)) {
            sc.inputs.erase(std::remove(sc.inputs.begin(), sc.inputs.end(), p),
                            sc.inputs.end());
            input_set.erase(p);
          }
          if(verbose >= 2) {
            msg("adding parent gate %s to subcircuit",
                p->get_var_name().c_str());
          }
          sc.nodes.push_back(p);
          node_set.insert(p);
          sc_set.insert(p);
          changed = true;
        }
      }
    }
  }

  // Keep nodes sorted by descending level
  std::sort(sc.nodes.begin(), sc.nodes.end(), [](const Gate* a, const Gate* b) {
    return a->get_level() > b->get_level();
  });
}
/*------------------------------------------------------------------------*/

static Subcircuit
gen_fsa_subcircuit(Gate* g) {
  Subcircuit result;

  for(int i = num_gates - 1; i >= 0; i--) {
    Gate* n = gates[i];

    if(n->get_level() > g->get_level())
      continue;

    if(!n->is_fsa() && !n->is_fsa_inp())
      continue;

    // FIX 1: Cutting FSA in pieces
    /* int constant = NN/2;


     if((n->get_cone_id() < g->get_cone_id()-constant) && n->get_cone_id() >= 0
     && g->get_cone_id() >= constant) continue;*/
    // FIX 1: ENDE

    if(n->get_input()) {
      result.inputs.push_back(n);
    } else {
      if(n->is_fsa_inp()) {
        result.inputs.push_back(n);
      } else {
        result.nodes.push_back(n);
      }
    }
  }

  ensure_parent_closure(result, g);

  std::sort(
    result.nodes.begin(), result.nodes.end(), [](const Gate* a, const Gate* b) {
      return a->get_level() > b->get_level();
    });

  // fix_missing_children(result);
  if(verbose >= 2)
    print_subcircuit(result);
  return result;
}
/*------------------------------------------------------------------------*/
static Subcircuit
gen_faha_subcircuit(const std::vector<std::pair<int, int>>& ids) {
  Subcircuit result;

  for(int i = num_gates - 1; i >= 0; i--) {
    Gate* n = gates[i];

    /*if(n->get_level() > g->get_level())
      continue;*/

    if(!n->has_any_faha_id() && !n->has_any_faha_inp_id())
      continue;

    // If this gate has any of the requested FA/HA ids and the type matches,
    // include it. This handles gates that belong to multiple instances.
    bool include = false;
    for(const auto& pr : ids) {
      int rid = pr.first;
      int rtype = pr.second;
      if(rid == -1)
        continue;
      if(n->has_faha_id_with_type(rid, rtype)) {
        include = true;
        break;
      }
    }
    if(include)
      result.nodes.push_back(n);
  }
  fix_missing_children(result);
  // ensure_parent_closure(result);
  std::sort(
    result.nodes.begin(), result.nodes.end(), [](const Gate* a, const Gate* b) {
      return a->get_level() > b->get_level();
    });

  // print_subcircuit(result);

  return result;
}

/*------------------------------------------------------------------------*/

// Extract a subcircuit by depth: include gates whose level is within
// `depth` levels below the root. Gates with level <= root_level - depth
// are treated as inputs. This function encapsulates the traversal logic
// previously embedded in `get_subcircuit` so callers can request a
// subcircuit by depth rather than by node count.
static Subcircuit
extract_subcircuit_by_depth(Gate* root, unsigned depth) {
  Subcircuit result;

  if(!root) {
    warning("no root gate provided for subcircuit extraction");
    return result;
  }

  // We'll perform a breadth-first traversal following child edges from the
  // root for `depth` levels. Collected nodes are in `result.nodes`. After
  // traversal, any child of a collected node that's not collected becomes
  // an input. Finally, run parent-closure to add parents whose both
  // children are present.

  std::unordered_set<Gate*> node_set;
  std::unordered_set<Gate*> input_set;

  // Adjust depth if root level is less than requested depth
  unsigned root_level = root->get_level();
  if(root_level < depth) {
    warning("root level %d is less than requested subcircuit depth %d, reset "
            "depth to %d",
            root_level,
            depth,
            root_level);
    depth = root_level;
  }

  // BFS levels starting from root
  std::vector<Gate*> curr_layer;
  curr_layer.push_back(root);
  node_set.insert(root);
  result.nodes.push_back(root);

  // Fallback: if the root is very close to inputs (small distance), include
  // all gates whose level is not above the root. This makes small-distance
  // roots expand to the complete cone below them instead of a depth-limited
  // traversal.
  /*if(root->get_dist() <= 6) {
    for(int i = num_gates - 1; i >= 0; --i) {
      Gate* n = gates[i];
      if(!n)
        continue;
      if(n->get_level() > root->get_level())
        continue;
      if(n->get_input()) {
        if(!input_set.count(n)) {
          input_set.insert(n);
          result.inputs.push_back(n);
        }
      } else {
        if(!node_set.count(n)) {
          node_set.insert(n);
          result.nodes.push_back(n);
        }
      }
    }

    // Add any parents whose both children are present and return.
    ensure_parent_closure(result, root);
    std::sort(result.nodes.begin(), result.nodes.end(), [](const Gate* a, const
  Gate* b) { return a->get_level() > b->get_level();
    });
    print_subcircuit(result);
    return result;
  }*/

  for(unsigned d = 0; d < depth; ++d) {
    std::vector<Gate*> next_layer;
    for(Gate* n : curr_layer) {
      // visit children
      unsigned ch0 = n->get_ch0();
      unsigned ch1 = n->get_ch1();

      if(ch0 >= 2) {
        Gate* c0 = gate(ch0);
        if(c0) {
          if(c0->get_input()) {
            if(!input_set.count(c0)) {
              input_set.insert(c0);
              result.inputs.push_back(c0);
            }
          } else if(!node_set.count(c0)) {
            node_set.insert(c0);
            next_layer.push_back(c0);
            result.nodes.push_back(c0);
          }
        }
      }
      if(ch1 >= 2) {
        Gate* c1 = gate(ch1);
        if(c1) {
          if(c1->get_input()) {
            if(!input_set.count(c1)) {
              input_set.insert(c1);
              result.inputs.push_back(c1);
            }
          } else if(!node_set.count(c1)) {
            node_set.insert(c1);
            next_layer.push_back(c1);
            result.nodes.push_back(c1);
          }
        }
      }
    }
    if(next_layer.empty()) {
      break;
    }
    curr_layer.swap(next_layer);
  }

  // For each collected node, any child not in node_set becomes an input
  for(Gate* n : result.nodes) {
    unsigned ch0 = n->get_ch0();
    unsigned ch1 = n->get_ch1();
    if(ch0 >= 2) {
      Gate* c0 = gate(ch0);
      if(c0 && !node_set.count(c0) && !input_set.count(c0)) {
        input_set.insert(c0);
        result.inputs.push_back(c0);
      }
    }
    if(ch1 >= 2) {
      Gate* c1 = gate(ch1);
      if(c1 && !node_set.count(c1) && !input_set.count(c1)) {
        input_set.insert(c1);
        result.inputs.push_back(c1);
      }
    }
  }

  // If the root's level equals the requested depth, but the collected
  // subcircuit is smaller than that level, we can try to expand by also
  // seeding traversal from gates in the neighboring (one smaller) COI.
  // This keeps the same distance/depth while adding nearby logic and avoids
  // evaluation failures that occur when the subcircuit is sparse.
  if(root->get_dist() <= depth) {
    // only expand when there is room to grow
    if(result.nodes.size() < static_cast<size_t>(root_level) - NN) {
      int root_coi = root->get_cone_id();
      // Iterate over several smaller COIs (neighboring cones) until the
      // subcircuit reaches the desired node count (bounded by root_level).
      for(int neighbor_coi = root_coi - 1;
          neighbor_coi >= 0
          && result.nodes.size() < static_cast<size_t>(root_level);
          --neighbor_coi) {

        if(verbose > 0) {
          msg("subcircuit fallback: seeding from neighbor COI %d (root COI %d)",
              neighbor_coi,
              root_coi);
        }

        // Collect candidate secondary roots: non-input gates in neighbor COI
        // with level not above the root. Prefer gates at the same level first.
        std::vector<Gate*> candidates;
        for(int i = num_gates - 1; i >= 0; --i) {
          Gate* g = gates[i];
          if(!g)
            continue;
          if(g->get_input())
            continue;
          if(g->get_cone_id() < neighbor_coi)
            continue;
          if(g->get_level() > root_level)
            continue;
          candidates.push_back(g);
        }

        // For each candidate, perform a depth-limited BFS and merge results.
        for(Gate* cand : candidates) {
          if(node_set.count(cand))
            continue;// already present
          std::vector<Gate*> s_curr;
          s_curr.push_back(cand);
          std::unordered_set<Gate*> local_seen;
          local_seen.insert(cand);
          // add candidate as a node
          node_set.insert(cand);
          result.nodes.push_back(cand);

          for(unsigned d = 0; d < depth; ++d) {
            std::vector<Gate*> s_next;
            for(Gate* n : s_curr) {
              unsigned ch0 = n->get_ch0();
              unsigned ch1 = n->get_ch1();
              if(ch0 >= 2) {
                Gate* c0 = gate(ch0);
                if(c0) {
                  if(c0->get_input()) {
                    if(!input_set.count(c0)) {
                      input_set.insert(c0);
                      result.inputs.push_back(c0);
                    }
                  } else if(!node_set.count(c0) && !local_seen.count(c0)) {
                    node_set.insert(c0);
                    local_seen.insert(c0);
                    s_next.push_back(c0);
                    result.nodes.push_back(c0);
                  }
                }
              }
              if(ch1 >= 2) {
                Gate* c1 = gate(ch1);
                if(c1) {
                  if(c1->get_input()) {
                    if(!input_set.count(c1)) {
                      input_set.insert(c1);
                      result.inputs.push_back(c1);
                    }
                  } else if(!node_set.count(c1) && !local_seen.count(c1)) {
                    node_set.insert(c1);
                    local_seen.insert(c1);
                    s_next.push_back(c1);
                    result.nodes.push_back(c1);
                  }
                }
              }
            }
            if(s_next.empty())
              break;
            s_curr.swap(s_next);
          }

          // stop seeding candidates once we've reached the size target
          if(result.nodes.size() >= static_cast<size_t>(root_level))
            break;
        }
      }
    }
  }

  // Ensure parents whose both children are present as inputs/nodes are added
  ensure_parent_closure(result, root);

  // Keep nodes sorted by descending level for downstream consumers
  std::sort(
    result.nodes.begin(), result.nodes.end(), [](const Gate* a, const Gate* b) {
      return a->get_level() > b->get_level();
    });

  if(verbose >= 1)
    print_subcircuit(result);
  return result;
}
/*------------------------------------------------------------------------*/
static std::string dump_circuit_name;        // empty => dumping disabled
static std::filesystem::path dump_dir;
static std::mutex dump_mutex;

void
set_subcircuit_dump(const std::string& circuit_file, const std::string& out_dir) {
  dump_circuit_name = std::filesystem::path(circuit_file).stem().string();
  dump_dir = std::filesystem::path(out_dir) / dump_circuit_name;
}

static std::string
json_escape(const std::string& s) {
  std::string r;
  r.reserve(s.size());
  for(char c : s) {
    switch(c) {
    case '"': r += "\\\""; break;
    case '\\': r += "\\\\"; break;
    case '\n': r += "\\n"; break;
    case '\t': r += "\\t"; break;
    default:
      if(static_cast<unsigned char>(c) < 0x20) {
        char buf[8];
        snprintf(buf, sizeof buf, "\\u%04x", c);
        r += buf;
      } else
        r += c;
    }
  }
  return r;
}

static void
write_gate_array(std::ofstream& out, const char* key,
                 const std::vector<Gate*>& gs) {
  out << "  \"" << key << "\": [";
  for(size_t i = 0; i < gs.size(); ++i) {
    Gate* g = gs[i];
    out << (i ? ",\n" : "\n") << "    {\"var\": " << g->get_var()
        << ", \"name\": \"" << json_escape(g->get_var_name())
        << "\", \"level\": " << g->get_level()
        << ", \"cone\": " << g->get_cone_id() << "}";
  }
  out << (gs.empty() ? "]" : "\n  ]");
}

void
save_subcircuit_to_file(Subcircuit& sc, Gate* root, unsigned size,
                        const char* mode) {
  if(dump_circuit_name.empty() || !root)
    return;

  std::lock_guard<std::mutex> lock(dump_mutex);

  std::error_code ec;
  std::filesystem::create_directories(dump_dir, ec);
  if(ec) {
    warning("cannot create directory %s: %s",
            dump_dir.string().c_str(), ec.message().c_str());
    return;
  }

  auto file = dump_dir / ("root_" + std::to_string(root->get_var()) + ".json");
  std::ofstream out(file);
  if(!out) {
    warning("cannot open %s for writing", file.string().c_str());
    return;
  }

  out << "{\n";
  out << "  \"circuit\": \"" << json_escape(dump_circuit_name) << "\",\n";
  out << "  \"root\": {\"var\": " << root->get_var() << ", \"name\": \""
      << json_escape(root->get_var_name()) << "\", \"level\": "
      << root->get_level() << "},\n";
  out << "  \"mode\": \"" << mode << "\",\n";
  out << "  \"depth\": " << size << ",\n";
  write_gate_array(out, "inputs", sc.inputs);
  out << ",\n";
  write_gate_array(out, "nodes", sc.nodes);
  out << "\n}\n";

  if(verbose > 0)
    msg("saved subcircuit for root %s to %s",
        root->get_var_name().c_str(), file.string().c_str());
}

static Subcircuit
get_subcircuit_impl(Gate* root, unsigned size, bool new_leading,
                    const char*& mode) {
  Subcircuit result;

  if(!root) {
    warning("no root gate provided for subcircuit extraction");
    return result;
  }
  if(root->get_input()) {
    warning("root gate %s is an input gate, cannot extract subcircuit",
            root->get_var_name().c_str());
    return result;
  }

  if(root->has_any_faha_maj_id() || root->has_any_faha_xor_id()) {
    mode = "faha";
    if(verbose > 0)
      msg("root gate %s is part of faha-substitute, extract faha subcircuit",
          root->get_var_name().c_str());
    return gen_faha_subcircuit(root->get_faha_ids());
  }

  if(root->is_fsa() && new_leading) {
    mode = "fsa";
    if(verbose > 0)
      msg("root gate %s , cone %d is part of fsa-substitute, extract fsa "
          "subcircuit",
          root->get_var_name().c_str(), root->get_cone_id());
    return gen_fsa_subcircuit(root);
  }

  mode = "depth";
  if(verbose > 0)
    msg("extracting subcircuit of depth %d from root gate %s with level %d, "
        "dist %d and COI %d",
        size, root->get_var_name().c_str(), root->get_level(),
        root->get_dist(), root->get_cone_id());

  return extract_subcircuit_by_depth(root, size);
}

Subcircuit
get_subcircuit(Gate* root, unsigned size, bool new_leading) {
  const char* mode = "none";
  Subcircuit sc = get_subcircuit_impl(root, size, new_leading, mode);
  if(!sc.nodes.empty())
    save_subcircuit_to_file(sc, root, size, mode);
  return sc;
}
/*------------------------------------------------------------------------*/

// Extracts a subcircuit of a given size from a root gate.
// The subcircuit consists of nodes (gates) and their input gates.
// Extracts a subcircuit of a given size from a root gate.
// The subcircuit consists of nodes (gates) and their input gates.
Subcircuit
get_subcircuit(Gate* root, unsigned size, bool new_leading) {
  Subcircuit result;

  if(!root) {
    warning("no root gate provided for subcircuit extraction");
    return result;// No root gate provided
  }

  if(root->get_input()) {
    warning("root gate %s is an input gate, cannot extract subcircuit",
            root->get_var_name().c_str());
    return result;// Root gate is an input, cannot extract subcircuit
  }

  // FA/HA-based substitute subcircuit
  if(root->has_any_faha_maj_id() || root->has_any_faha_xor_id()) {
    if(verbose > 0)
      msg("root gate %s is part of faha-substitute, extract faha subcircuit",
          root->get_var_name().c_str());
    return gen_faha_subcircuit(root->get_faha_ids());
  }

  // FSA-based substitute subcircuit
  if(root->is_fsa() && new_leading) {
    if(verbose > 0)
      msg("root gate %s , cone %d is part of fsa-substitute, extract fsa "
          "subcircuit",
          root->get_var_name().c_str(),
          root->get_cone_id());
    return gen_fsa_subcircuit(root);
  }
  // Interpret `size` as depth (levels) below the root and delegate to helper.
  if(verbose > 0)
    msg(
      "extracting subcircuit of depth %d from root gate %s with level %d, dist "
      "%d and COI %d",
      size,
      root->get_var_name().c_str(),
      root->get_level(),
      root->get_dist(),
      root->get_cone_id());

  return extract_subcircuit_by_depth(root, size);
}
/*------------------------------------------------------------------------*/

/*------------------------------------------------------------------------*/
// Randomly assigns values to the inputs of a subcircuit
static void
sample_subcircuit_random(Subcircuit& sc) {
  if(sc.inputs.empty()) {
    warning("subcircuit has no inputs, cannot sample");
    return;
  }

  // Randomly assign true and false values to inputs
  for(auto& input : sc.inputs) {
    sample_t value = sample_uint64();
    input->set_value(value);
    LOGDBG("  sampling input %s with value %llu",
           input->get_var_name().c_str(),
           value);
  }
}
/*------------------------------------------------------------------------*/
// Assigns a uniform matrix of bitvectors to the inputs of a subcircuit for a
// given batch
static void
sample_subcircuit_uniform_matrix(Subcircuit& sc, unsigned batch) {
  if(sc.inputs.empty()) {
    warning("subcircuit has no inputs, cannot sample");
    return;
  }

  LOGDBG("sampling subcircuit with uniform matrix for batch %u", batch);

  // reset all inputs to 0
  for(auto& input : sc.inputs) {
    input->set_value(0);
  }

  size_t start = batch * batch_size;
  size_t end = std::min(start + batch_size, sc.inputs.size());

  // in the batch we want to assign the bitvectors 2^63, 2^62, ..., 2^0
  for(size_t i = start; i < end; ++i) {
    size_t bit_pos = 63 - (i - start);// 2^63 for first, 2^62 for second, etc.
    sample_t value = (sample_t(1) << bit_pos);
    sc.inputs[i]->set_value(value);
    LOGDBG("  uniform matrix: input %s gets value %llu (bit %zu set)",
           sc.inputs[i]->get_var_name().c_str(),
           value,
           bit_pos);
  }
}

/*------------------------------------------------------------------------*/
// Evaluates all gates in the subcircuit
static void
eval_subcircuit(Subcircuit& sc) {
  for(auto it = sc.nodes.rbegin(); it != sc.nodes.rend(); ++it) {
    Gate* node = *it;
    if(node->get_input()) {
      LOGDBG("skipping input gate %s", node->get_var_name().c_str());
      continue;
    }
    node->eval_gate();
  }
}
/*------------------------------------------------------------------------*/
void
sample_full_truth_table(Subcircuit& sc, nmod_mat_t mat) {
  LOGDBG("sampling full truth table for subcircuit with %zu inputs",
         sc.inputs.size());

  int n = sc.inputs.size();
  uint64_t rows = 1ULL << n;
  uint64_t num_batches = (rows + batch_size - 1) / batch_size;

  nmod_mat_init(mat, rows, get_size(sc) + 1, 42);

  // set first column to constant 1
  set_constant_column(mat, 0);

  // iterate over batches
  for(uint64_t batch = 0; batch < num_batches; batch++) {
    auto input_values = make_bit_pattern_batch(n, batch);

    // assign input values for this batch
    for(int i = 0; i < n; i++)
      sc.inputs[i]->set_value(input_values[i]);

    // evaluate subcircuit
    eval_subcircuit(sc);

    // insert inputs into matrix
    uint64_t col_idx = 1;
    for(auto it = sc.inputs.rbegin(); it != sc.inputs.rend(); ++it) {
      sample_t s = (*it)->get_value();
      mat_insert_batch(mat, batch * batch_size, col_idx, s, rows);
      col_idx++;
    }

    // insert internal nodes into matrix
    for(auto it = sc.nodes.rbegin(); it != sc.nodes.rend(); ++it) {
      sample_t s = (*it)->get_value();
      mat_insert_batch(mat, batch * batch_size, col_idx, s, rows);
      col_idx++;
    }
  }
}
/*------------------------------------------------------------------------*/
static void
random_sampling(Subcircuit& sc, nmod_mat_t mat, int sample_size) {

  // Random sampling: assign random values to inputs and evaluate
  for(unsigned k = 0; k <= sample_size / batch_size; ++k) {
    sample_subcircuit_random(sc);// Sample new values for inputs
    eval_subcircuit(sc);         // Evaluate the subcircuit with sampled inputs

    // set values in matrix
    uint64_t j = 1;
    for(auto it = sc.inputs.rbegin(); it != sc.inputs.rend(); ++it) {
      sample_t s = (*it)->get_value();
      mat_insert_batch(mat, batch_size * k, j, s, nmod_mat_nrows(mat));
      j++;
    }
    for(auto it = sc.nodes.rbegin(); it != sc.nodes.rend(); ++it) {
      sample_t s = (*it)->get_value();
      mat_insert_batch(mat, batch_size * k, j, s, nmod_mat_nrows(mat));
      j++;
    }
  }
}
/*------------------------------------------------------------------------*/

static void
weighted_sampling(Subcircuit& sc, nmod_mat_t mat, int sample_size) {

  int idx = 0;
  std::unordered_map<int, int> var_to_dimacs;
  for(auto it = sc.inputs.rbegin(); it != sc.inputs.rend(); ++it) {
    int v = (*it)->get_var() >> 1;
    var_to_dimacs[v] = idx;
    idx++;
  }
  for(auto it = sc.nodes.rbegin(); it != sc.nodes.rend(); ++it) {
    int v = (*it)->get_var() >> 1;
    var_to_dimacs[v] = idx;
    idx++;
  }

  if(mat->r == 0)
    return;

  // Fixed on purpose: `start_row` below is used as the CMSGen RNG seed, so the
  // block layout fully determines the sample matrix. Deriving it from the pool's
  // thread count made the sampling -- and therefore the entire guess-and-prove
  // computation -- depend on the number of cores of the machine.
  const int nr_blocks = std::min<int>(SAMPLE_BLOCKS, static_cast<int>(mat->r));
  int block_size = mat->r / nr_blocks;
  int remainder = mat->r % nr_blocks;
  int start_row = 0;

  for(int t = 0; t < nr_blocks; t++) {
    int rows_for_thread = block_size + (t < remainder ? 1 : 0);
    int end_row = start_row + rows_for_thread;

    // Capture var_to_dimacs by reference (it's local and will outlive all tasks
    // due to pool->wait())
    auto task = [&sc, &mat, &var_to_dimacs, start_row, end_row]() {
      uint32_t seed = start_row;
      CMSGen::SATSolver solver(nullptr, nullptr, &seed);
      solver.new_vars(sc.inputs.size() + sc.nodes.size());

      std::vector<CMSGen::Lit> clause;
      for(const auto& gate : sc.nodes) {
        for(int l : gate->get_cnf_encoding()) {
          if(l == 0) {
            solver.add_clause(clause);
            clause.clear();
          } else if(l > 0)
            clause.emplace_back(var_to_dimacs[abs(l)], false);
          else
            clause.emplace_back(var_to_dimacs[abs(l)], true);
        }
      }

      for(int i = start_row; i < end_row; i++) {
        solver.solve();
        auto model = solver.get_model();
        for(uint32_t var = 0; var < solver.nVars(); var++) {
          int val = (model[var] == CMSGen::l_True) ? 1 : 0;
          nmod_mat_set_entry(mat, i, var + 1, val);
        }
      }
    };
    pool->detach_task(task);
    start_row = end_row;
  }
  pool->wait();
}
/*------------------------------------------------------------------------*/
// Samples and evaluates a subcircuit using both uniform matrix and random
// sampling
extern bool uniform_sampling;
void
sample_and_eval_subcircuit(Subcircuit& sc, unsigned samples, nmod_mat_t mat) {
  if(sc.nodes.empty()) {
    warning("subcircuit is empty, nothing to evaluate");
    return;
  }
  if(sc.inputs.empty()) {
    warning("subcircuit has no inputs, cannot evaluate");
    return;
  }

  // Ensure deterministic ordering of nodes and inputs: nodes by descending
  // level and inputs by their appearance (stable). Several code paths rely
  // on a consistent ordering when mapping model values to matrix columns.
  std::sort(sc.nodes.begin(), sc.nodes.end(), [](const Gate* a, const Gate* b) {
    return a->get_level() > b->get_level();
  });
  // keep inputs stable but sort by level as well to be deterministic
  std::sort(
    sc.inputs.begin(), sc.inputs.end(), [](const Gate* a, const Gate* b) {
      return a->get_level() > b->get_level();
    });

  // here we use a dummy modulus because it does not matter
  nmod_mat_init(mat, samples, get_size(sc) + 1, 42);

  
  // set first column to constant 1
  set_constant_column(mat, 0);
  if(uniform_sampling)
    weighted_sampling(sc, mat, samples);
  else
    random_sampling(sc, mat, samples);


  // reset_subcircuit(sc);
}
/*------------------------------------------------------------------------*/
// Resets all values in the subcircuit to unknown/initial state
inline void
reset_subcircuit(Subcircuit& sc) {
  if(sc.nodes.empty()) {
    warning("subcircuit is empty, nothing to reset");
    return;
  }

  LOGDBG("resetting subcircuit with %zu nodes", sc.nodes.size());
  for(auto& node : sc.nodes) {
    node->reset_value();
    LOGDBG("  reset gate %s to unknown value", node->get_var_name().c_str());
  }

  for(auto& input : sc.inputs) {
    input->reset_value();
    LOGDBG("  reset input %s to unknown value", input->get_var_name().c_str());
  }
}
/*------------------------------------------------------------------------*/
void
get_cnf_encoding(Subcircuit& sc, std::vector<int>& cnf) {
  cnf.clear();
  for(const auto& gate : sc.nodes) {
    auto gate_cnf = gate->get_cnf_encoding();
    cnf.insert(cnf.end(), gate_cnf.begin(), gate_cnf.end());
  }
}
/*------------------------------------------------------------------------*/

void
print_subcircuit(Subcircuit& sc) {
  if(sc.nodes.empty()) {
    warning("subcircuit is empty, nothing to print");
    return;
  }

  std::cout << "Subcircuit with " << sc.nodes.size() << " nodes and "
            << sc.inputs.size() << " inputs:" << std::endl;

  std::cout << "Inputs:" << std::endl;
  for(auto& input : sc.inputs) {
    std::cout << "  " << input->get_var_name()
              << ",    level= " << input->get_level()
              << ",    cone= " << input->get_cone_id() << std::endl;
  }

  std::cout << "Nodes:" << std::endl;
  for(auto& node : sc.nodes) {
    std::cout << "  " << node->get_var_name()
              << ",    level= " << node->get_level()
              << ",    cone= " << node->get_cone_id() << std::endl;
  }
}
/*------------------------------------------------------------------------*/

}// namespace talisman
