# This is the default target, which will be built when 
# you invoke make
.PHONY: all
all: HelloWorld

# This rule tells make how to build hello from hello.cpp
HelloWorld: HelloWorld.cpp
	g++ HelloWorld.cpp -o HelloWorld -g -O3 -march=native 

# This rule tells make to copy hello to the binaries subdirectory,
# creating it if necessary
.PHONY: install
install:
	mkdir -p binaries
	cp -p HelloWorld binaries

# This rule tells make to delete hello and hello.o
.PHONY: clean 
clean:
	rm -f HelloWorld