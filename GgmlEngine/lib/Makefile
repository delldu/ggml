#/************************************************************************************
#***
#***	Copyright 2024 Dell(18588220928@163.com), All Rights Reserved.
#***
#***	File Author: Dell, Tue 30 Jan 2024 11:53:11 PM CST
#***
#************************************************************************************/
#

LIBNAME := libggml_engine

# Installation related variables and target
INSTALL_PATH ?= $(shell pkg-config ggml --cflags | sed s/-I//g)

INCS	:=  \
		$(shell pkg-config nimage --cflags) \
		$(shell pkg-config ggml --cflags) \
		-Ilib/include

SOURCE :=  \
	source/ggml_engine.c

DEFINES := \
 	-DGGML_CUDA \
 	-D_GLIBCXX_USE_CXX11_ABI=1 -std=c++11 -std=gnu++11


CFLAGS := -O3 -fPIC -Wall -Wextra
CXXFLAGS := ${CFLAGS}
LDFLAGS := -fPIC


#****************************************************************************
# Makefile code common to all platforms
#****************************************************************************
CFLAGS   := ${CFLAGS} ${DEFINES}
CXXFLAGS := ${CXXFLAGS} ${DEFINES}
OBJECTS := $(addsuffix .o,$(basename ${SOURCE}))

#****************************************************************************
# Compile block
#****************************************************************************
all: staticlib

staticlib:$(OBJECTS)
	$(AR) $(ARFLAGS) $(LIBNAME).a $(OBJECTS)


#****************************************************************************
# Depend block
#****************************************************************************
depend:

#****************************************************************************
# common rules
#****************************************************************************
%.o : %.cpp
	${CXX} ${CXXFLAGS} ${INCS} -c $< -o $@

%.o : %.c
	${CXX} ${CXXFLAGS} ${INCS} -c $< -o $@


clean:
	rm -rf *.a *.so *.o $(OBJECTS)

install:
	sudo mkdir -p ${INSTALL_PATH}
	sudo cp include/ggml_engine.h ${INSTALL_PATH}


#cmake -DGGML_CUBLAS=ON -DBUILD_SHARED_LIBS=OFF -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.1/bin/nvcc ..
