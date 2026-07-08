# .o 閺傚洣娆㈢€涙ɑ鏂佺捄顖氱窞
CC      := mpiicpc
# CC      := mpic++
# CFLAGS  := -O3
CFLAGS := -O3 -xSKYLAKE-AVX512 -qopt-zmm-usage=high

D_OBJ = ./obj
D_SRC = ./src

# TARGET = Solver_SHIzheng
# BIN_TAGRET = $(D_BIN)/$(TARGET)

SRC = $(wildcard $(D_SRC)/*.cpp)
OBJ = $(addprefix $(D_OBJ)/, $(patsubst %.cpp, %.o, $(notdir $(SRC))))

VPATH = $(D_SRC)
vpath %.o $(D_OBJ)

# all: $(TARGET)
# Just change MPI_INCLUDE and MPI_LIB to the path of your mpi
MPI_INCLUDE := -I/share/intel/2018u4/compilers_and_libraries_2018.5.274/linux/mpi/include64
MPI_LIB := -L/share/intel/2018u4/compilers_and_libraries_2018.5.274/linux/mpi/lib64
MPI     := -lmpi

METIS_INCLUDE := -I/work/mae-huangt1/test/LIB/metis_lib/include
METIS_LIB :=  -L/work/mae-huangt1/test/LIB/metis_lib/lib
METIS     := -lmetis

GK_INCLUDE := -I/work/mae-huangt1/test/LIB/gk_lib/include
GK_LIB := -L/work/mae-huangt1/test/LIB/gk_lib/lib
GK	   := -lGKlib

EXEC    := main_mpiDSMC_ht

$(EXEC) : $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(METIS_INCLUDE) $(METIS_LIB) $(METIS) $(GK_INCLUDE) $(GK_LIB) $(GK)
	# rm -rf $(OBJ)

$(D_OBJ)/%.o : %.cpp
	$(CC) $(CFLAGS) -c $< -o $@ $(METIS_INCLUDE) $(METIS_LIB) $(METIS) $(GK_INCLUDE) $(GK_LIB) $(GK)

.PHONY : clean echoSRC echoOBJ cleanAll
clean :
	rm -rf $(OBJ)
echoSRC :
	@echo $(SRC)
echoOBJ :
	@echo $(OBJ)
cleanAll :
	rm $(OBJ) $(EXEC)