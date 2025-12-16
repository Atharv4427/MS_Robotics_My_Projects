
	extern int orders, *qty, *orderlen;
	extern double len,**initialptrn;
	extern double EPSILON;

	extern int *rmatind;

	extern CPXENVptr envm ,envs;
	extern CPXLPptr lpm, lps;

	extern void read_input_data();
	extern int get_number_of_digits(int a);
	extern	void error(char *x);

	extern void generate_initial_patterns();
	extern void setup_master_sub_problems();
	extern void solve_column_generation();
	