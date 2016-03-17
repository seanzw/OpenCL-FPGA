
CLC = sdaccel
CC = touch

KERNELDIR = kernel
OUTPUTDIR = FPGA

TARGETS = host
TARGETS += conv1_baseline conv1_workgroup conv1_unroll conv1_pipeline conv1_item_pipeline

TCLS = $(addsuffix .tcl, $(TARGETS))
OBJS = $(addsuffix .o, $(TARGETS))

TCLFILES = $(addprefix, $(KERNELDIR)/, $(TCLS))
OBJFILES = $(addprefix, $(OUTPUTDIR)/, $(OBJS))

BINDIR = pkg/pcie
BIN = $(BINDIR)/alpha.xclbin
HOST = ./$(OUTPUTDIR)/host/$(BINDIR)/host.exe


all: $(TCLFILES) $(OBJFILES)

$(OBJS):
	$(CLC) $(KERNELDIR)/$(@:.o=.tcl)
	$(CC) $(OUTPUTDIR)/$@

$(TARGETS): $(OUTPUTDIR)/host.o
	$(HOST) $(KERNELDIR)/$@.xml $(KERNELDIR)/result_$@.xml $(OUTPUTDIR)/$@/$(BIN)
	
clean: 
	rm -rf $(addprefix $(OUTPUTDIR)/, $(TARGETS))
	rm $(OBJFILES)
