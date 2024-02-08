# lfts_gpu
Langevin Field-Theoretic Simulation of Diblock Copolymers on the GPU

See https://tbeardsley.com/projects/lfts/fts_gpu for a detailed discussion of this project.

Input file format:

Line1: N NA XN C Ndt isXeN
Line2: mx my mz Lx Ly Lz

Line3: n_eq n_st n_smpl loadType

Lines 4->(M+3): W-(r)

Lines (M+4)->(2M+3): w+(r)

Parameter descriptions:
N is the number of monomers in a single polymer chain (integer).
NA is the number of monomers in the A-block of a polymer chain (integer).
XN is the interaction strength between A and B-type monomers (double).
C is the square root of the invariant polymerisation index, Nbar (double).
Ndt is the size of the time step in the Langevin update of W-(r) (double).
isXeN instructs the program whether the parameter XN is in terms of bare (isXeN=0) or effective (isXeN=1) chi (integer).
mx, my, mz are the number of mesh points in the x, y, and z dimensions of the simulation box (integers).
Lx, Ly, Lz are the dimensions of the simulation box (in units of the polymer end-to-end length, R0) in the x, y, and z dimensions (doubles).
n_eq is the number of langevin steps performed to equilibrate the system (integer).
n_st is the number of langevin steps performed after equilibration has ended, during which statistics are sampled (integer).
n_smpl is the number of steps between samples taken in the statistics period (integer).
loadType instructs the program whether to load the W-(r) and w+(r) fields from the proceeding file lines (loadType=1), start from a disordered state (loadType=0) or start from a (300) lamellar phase (loadType=2).
M = mx*my*mz is the total number of mesh points, such that the proceeding 2*M lines of the file can hold W-(r) and w+(r) fields to load.

