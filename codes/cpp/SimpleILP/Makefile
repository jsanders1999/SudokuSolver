# This is the default target, which will be built when 
# you invoke make
.PHONY: all
all: highs

# This rule tells make how to build hello from hello.cpp
highs: call_highs_from_cpp.cpp
	g++ -o call_highs_from_cpp -I ./HiGHS/src/ call_highs_from_cpp.cpp 

# This rule tells make to copy hello to the binaries subdirectory,
# creating it if necessary
.PHONY: install
install:
	mkdir -p binaries
	cp -p call_highs_from_cpp binaries

# This rule tells make to delete hello and hello.o
.PHONY: clean 
clean:
	rm -f call_highs_from_cpp