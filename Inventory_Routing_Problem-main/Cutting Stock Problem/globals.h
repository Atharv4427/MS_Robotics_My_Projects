
int orders=3, *qty=NULL, *orderlen=NULL;
double len=10.0, **initialptrn=NULL;

double EPSILON = 0.005;

int *rmatind=NULL;

CPXENVptr envm = NULL, envs = NULL;
CPXLPptr lpm = NULL, lps = NULL;
