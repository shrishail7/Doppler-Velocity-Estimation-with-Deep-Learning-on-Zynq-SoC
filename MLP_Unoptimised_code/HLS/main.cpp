#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mlp_run.h"
#include "main.h"

void MLP(hls::stream<axis_data> &Input, hls::stream<axis_data> &Output)
{
	 mlp_init();

// Stream Interface
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INTERFACE axis register both port=Input
#pragma HLS INTERFACE axis register both port=Output

	 // Declaring Input and Output
Data_type InputA[Input_Size];
Data_type OutputY[Output_Size];

// internal variables for stream data type
axis_data local_read , local_write;

	// taking inputs from stream and converting to normal to execute mlp_forward function
	for(int i=0;i<Input_Size;i++){

		local_read = Input.read();
		InputA[i] = local_read.data;
	}

	// calling MLP_forward function to generate output
	mlp_forward(InputA, OutputY);

	// convert output generated to stream
	for(int i=0;i<Output_Size;i++){

		local_write.data = OutputY[i];
		local_write.keep = -1;
		if(i==Output_Size-1){
			local_write.last = 1;
		}
		else{
			local_write.last = 0;
		}
		Output.write(local_write);
	}

}
