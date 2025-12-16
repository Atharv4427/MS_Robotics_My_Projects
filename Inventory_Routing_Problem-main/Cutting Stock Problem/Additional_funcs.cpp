#include "master.h" 
#include "externs.h"


void read_input_data()
{
	char s[100];
	int i,position;

	printf("Reading input file file");

	ifstream fin("input.txt");

	if (fin.fail())
	{																//Check for Error
		cerr << "Error Opening File input.txt";
		exit(1);
	}

	fin.getline(s, 100);                                   //length of stock
	for (i = 0;s[i] != ':';i++)
		;
	len = atof(&s[i+1]);

	fin.getline(s, 100);                                   //number of orders
	for (i = 0;s[i] != ':';i++)
		;
	orders = atoi(&s[i+1]);
	printf("\n\nLength of stock: %.2f \nNumber of orders: %d\n",len,orders);

	orderlen = new int[orders];
	qty = new int[orders];

	fin.getline(s, 100);  

	for(i=0;i<orders;i++)
	{
		fin.getline(s, 100);

		orderlen[i] = atoi(&s[0]);

		position = get_number_of_digits(orderlen[i]) + 1;

		qty[i] = atoi(&s[position]);
		printf("\nOrder %d: Length: %d, Qty: %d",i+1,orderlen[i],qty[i]);
	}

	fin.close();
}

int get_number_of_digits(int a)
{
	int i=0;

	if(a==0)
		return 1;

	while(a>0)
	{
		a /= 10;
		i++;
	}

	return i;
}

void generate_initial_patterns()
{
	int i,j;

	initialptrn = new double*[orders];

	for(i=0;i<orders;i++)
	{
		initialptrn[i] = new double[orders];

		for(j=0;j<orders;j++)
			if(i==j)
				initialptrn[i][j] = floor(len/orderlen[i]);
			else
				initialptrn[i][j] = 0.0;
	}

	printf("\n\nInitial Patterns::");
	for(i=0;i<orders;i++)
	{
		printf("\n");
		for(j=0;j<orders;j++)
			printf("%.1f\t",initialptrn[i][j]);
	}


}

void setup_master_sub_problems()
{
	int i,status,rmatbeg[1];
	double *obj, *cons, rhs[1];
	char *ctype, sign[1];
	
	//setup Master Problem
	envm = CPXopenCPLEX(&status);
	if(envm == NULL)
		error("envm is null");

	// create LP problem (Master LP)
	lpm = CPXcreateprob(envm,&status,"Masterprob");	
	if(lpm == NULL)
		error("lpm is null");

	// declare it as a minimization problem
	CPXchgobjsen(envm,lpm,CPX_MIN);

	rmatind =  new int[orders];
	obj = new double[orders];
	ctype = new char[orders];
	cons = new double[orders];

	for(i=0;i<orders;i++)
	{
		rmatind[i] = i;
		obj[i] = 1.0;
		ctype[i] = 'C';
	}
	
	CPXnewcols(envm,lpm,orders,obj,NULL,NULL,ctype,NULL);

	sign[0] = 'G';
	rmatbeg[0] = 0;

	for(i=0;i<orders;i++)
	{
		rhs[0] = qty[i];
		CPXaddrows(envm,lpm,0,1,orders,&rhs[0],&sign[0],rmatbeg,rmatind,initialptrn[i],NULL,NULL);
	}

	CPXwriteprob(envm,lpm,"MasterProblem.txt","LP");
	printf("\n\nMaster problem written to MasterProblem.txt");


	//setup Sub Problem
	envs = CPXopenCPLEX(&status);
	if(envs == NULL)
		error("envs is null");

	// create LP problem (Sub LP)
	lps = CPXcreateprob(envs,&status,"Subprob");	
	if(lps == NULL)
		error("lps is null");

	// declare it as a Max problem
	CPXchgobjsen(envs,lps,CPX_MAX);

	for(i=0;i<orders;i++)
	{
		ctype[i] = 'I';
		cons[i] = orderlen[i];
	}

	CPXnewcols(envs,lps,orders,obj,NULL,NULL,ctype,NULL);
	CPXchgprobtype(envs,lps,1);

	sign[0] = 'L';
	rhs[0] = len;
	CPXaddrows(envs,lps,0,1,orders,&rhs[0],&sign[0],rmatbeg,rmatind,cons,NULL,NULL);

	CPXwriteprob(envs,lps,"SubProblem.txt","LP");
	printf("\nSub problem written to SubProblem.txt");

	free(obj);
	free(ctype);
	free(cons);	
}

void solve_column_generation()
{
	int i,status,cmatbeg[1];
	double objvallp, objvalip, *lpsoln, *newpatrn, *pi,obj[1];

	newpatrn = new double[orders];
	pi = new double[orders];
	lpsoln = new double[orders+100];
	
	obj[0] = 1.0;
	cmatbeg[0] = 0;

	CPXchgprobtype(envm,lpm,0);

	CPXchgprobtype(envs,lps,1);
	//CPXsetintparam(envs,CPX_PARAM_SCRIND,1);

	while(1)
	{
		CPXprimopt(envm,lpm);
		CPXsolution(envm,lpm,&status,&objvallp,lpsoln,pi,NULL,NULL);
		printf("\n\nCurrent Master prob objval: %.3f",objvallp);	//\nStatus: %d ,status
		/*for(i=0;i<orders;i++)
			printf("\nx%d: %.3f",i+1,lpsoln[i]);*/

		
		CPXchgobj(envs,lps,orders,rmatind,pi);
		//CPXwriteprob(envs,lps,"SubProblem.txt","LP");
		//printf("\nSub problem written to SubProblem.txt");
		//getch();

		CPXmipopt(envs,lps);
		CPXgetmipobjval(envs,lps,&objvalip);
		printf("\n\nCurrent Sub prob objval: %.3f",objvalip);

		if((1.0-objvalip)>EPSILON || fabs(1.0-objvalip)<EPSILON)
			break;

		printf("\nNew Pattern Generated...");
		CPXgetmipx(envs,lps,newpatrn,0,orders-1);

		CPXaddcols(envm,lpm,1,orders,obj,cmatbeg,rmatind,newpatrn,NULL,NULL,NULL);	
		//CPXwriteprob(envm,lpm,"MasterProblem.txt","LP");
		//printf("\nUpdated Master problem written to MasterProblem.txt");
		getch();
	}

	CPXwriteprob(envm,lpm,"MasterProblem.txt","LP");
	printf("\nProblem Solved..");
}


void error(char *x)
{
	printf("\n\n%s\n",x);
	_getch();
	exit(0);
}