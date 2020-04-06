CXXFLAGS = -I include  -std=c++11 -O3 -I/root/anaconda3/envs/python367/include/python3.6m
LDFLAGS = $(shell python3-config --ldflags)

DEPS = $(shell find include -xtype f)
CXX_SOURCES = dist.cpp
OPENCV = `pkg-config --cflags --libs opencv`

LIB_SO = dist.so

$(LIB_SO): $(CXX_SOURCES) $(DEPS)
	$(CXX) -o $@ $(CXXFLAGS) $(LDFLAGS) $(CXX_SOURCES) --shared -fPIC $(OPENCV)

clean:
	rm -rf $(LIB_SO)
