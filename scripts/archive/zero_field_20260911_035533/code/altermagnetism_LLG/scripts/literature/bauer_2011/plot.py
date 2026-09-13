"""Plot all saved trajectory summaries in reduced units."""
from scripts.literature.workflow import analysis_arguments,plot_trajectory


if __name__=='__main__':
    args=analysis_arguments()
    plot_trajectory(args.input,args.output)

