
#include "master.h"
#include "globals.h"
#include "externs.h"



void main()
{

	read_input_data();
	
	generate_initial_patterns();
	
	setup_master_sub_problems();

	solve_column_generation();

	getch();
}


