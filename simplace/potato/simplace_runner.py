import simplace
import jpype
import yaml
import sys

def run_simplace(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        config = config['local']

    install_dir = config['install_dir']
    work_dir = config['work_dir']
    output_dir = config['output_dir']
    solution_file = config['solution_file']
    project_file = config['project_file']
    
    sp = simplace.initSimplace(install_dir, work_dir, output_dir)

    simplace.openProject(sp, solution_file, project_file)
    simplace.runProject(sp)
    sp.shutDown()
    jpype.java.lang.System.exit(0)

if __name__ == "__main__":                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
    if len(sys.argv) != 2:
        print("Usage: python simplace_run.py <config.yaml>")
        sys.exit(1)
    run_simplace(sys.argv[1])
