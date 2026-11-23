/*------------------------------------------------------------------------*/
/*! \file parser.h
    \brief contains functions necessary to parse the AIG

  Part of TalisMan
  Copyright(C) 2025 TalisMan-Developers
*/
/*------------------------------------------------------------------------*/
#ifndef TALISMAN_SRC_SUBCIRCUIT_H_
#define TALISMAN_SRC_SUBCIRCUIT_H_
/*------------------------------------------------------------------------*/
#include <nmod_mat.h>
#include <unordered_set>
#include <vector>
#include <string>

#include "gate.hpp"
#include "linear_algebra.hpp"
#include "linpolynomial.hpp"
/*------------------------------------------------------------------------*/

namespace talisman {

void
seed_randomness();

struct Subcircuit {
  std::vector<Gate*> nodes;
  std::vector<Gate*> inputs;
};

Subcircuit
get_subcircuit(Gate* root, unsigned size, bool new_leading);

// Alternative selection: include gates from the same or smaller COI (cone id)
// whose level is at most the level of the root gate. The `size` parameter
// still bounds how far below the root level to include gates (levels <
// root_level - size become inputs).
Subcircuit
get_subcircuit_by_coi(Gate* root, unsigned size);

inline size_t
get_size(Subcircuit& sc) {
  return sc.inputs.size() + sc.nodes.size();
};

void
sample_full_truth_table(Subcircuit& sc, nmod_mat_t mat);

void
sample_and_eval_subcircuit(Subcircuit& sc, unsigned samples, nmod_mat_t mat);

inline void
reset_subcircuit(Subcircuit& sc);

inline Gate*
get_gate(Subcircuit& sc, size_t i) {
  if(i == 0)
    return NULL;
  else if(i <= sc.inputs.size())
    return sc.inputs[sc.inputs.size() - i];
  else
    return sc.nodes[sc.nodes.size() - (i - sc.inputs.size())];
}

void
print_subcircuit(Subcircuit& sc);
// Enable dumping: files go to <out_dir>/<circuit_stem>/root_<var>.json.
// If this is never called, nothing is written.
void
set_subcircuit_dump(const std::string& circuit_file,
                    const std::string& out_dir = "subcircuits");

void
save_subcircuit_to_file(Subcircuit& sc, Gate* root, unsigned size,
                        const char* mode);
}// namespace talisman

#endif// TALISMAN_SRC_SUBCIRCUIT_H_
