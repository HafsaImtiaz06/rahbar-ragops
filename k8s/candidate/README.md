The three files generated here are produced by `scripts/f3_release_demo.sh` (run from the repo root):

- `indexer-candidate.yaml` (step `candidate-index`): the indexer Job building the candidate collection (base name `cand`)
- `api-candidate.yaml` (step `candidate-up`): second API Deployment + its own Service `api-candidate`
- `eval-candidate.yaml` (step `eval-candidate`): M2's eval Job copied to target `api-candidate`, results in `/results/candidate`

Regenerate them, do not hand-edit. The candidate collection base name must never start with `rahbar-`
(the live indexer prunes every `rahbar-*` collection that is not active or previous).
