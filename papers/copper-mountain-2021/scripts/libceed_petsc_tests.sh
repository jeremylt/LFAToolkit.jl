# Echo command without running the command
function quoted_echo()
{
   local arg= string=
   for arg; do
      if [[ -z "${arg##* *}" ]]; then
         string+=" \"${arg//\"/\\\"}\""
      else
         string+=" $arg"
      fi
   done
   printf "%s\n" "${string# }"
}

# Run test over range of values
function run_tests()
{
  local mpi_run=($PETSC_DIR/$PETSC_ARCH/bin/mpiexec -n 8)
  local common_args="-ksp_type richardson -ksp_monitor -dm_plex_box_lower -3,-3,-3 -dm_plex_box_upper 3,3,3 -smoothing chebyshev"
  echo $common_args
  local max_p=6
  local max_v=4
  local num_dofs_1d=200
  # Fine grid
  local fine_p=
  for ((fine_p = 2; fine_p <= max_p; fine_p++)); do
    echo
    echo "fine p:     " $fine_p
    local num_cells_1d=$((num_dofs_1d/(fine_p-1)+1))
    # Coarse grid
    local coarse_p=
    for ((coarse_p = 1; coarse_p < fine_p; coarse_p++)); do
      echo
      echo "  coarse p: " $coarse_p
      # Smoothing passes
      local v=
      for ((v = 1; v <= max_v; v++)); do
        echo
        echo "    v:      " $v
        local all_args=("${common_args}" -degree $fine_p -coarse_degree $coarse_p -num_smooths $v -cells $num_cells_1d,$num_cells_1d,$num_cells_1d)
        echo 
        echo "Running test:"
        quoted_echo $mpi_run ./multigrid "${all_args[@]}"
        $mpi_run ./multigrid "${all_args[@]}" || \
          printf "\n\nError in test case; error code: $?\n\n"
      done
    done
  done
}
