import sys
import time
from multiprocessing import Process
from data_keeper import get_data_keeper
from run_experiment import init_common
from run_model_experiment import run_model
from run_selector_model_experiment import run_selector_model
from run_frn_model_experiment import run_frn_model
from run_extender_selector_model_experiment import run_extender_selector_model
from run_extender_frn_model_experiment import run_extender_frn_model
from run_boruta_model_experiment import run_boruta_model
from run_extender_robust_model_experiment import run_extender_robust_model


if __name__ == "__main__":
    drug = get_data_keeper().get_possible_second_level_drugs()[int(sys.argv[1])]
    init_common()
    processes = list()
    processes.append(Process(target=run_model, args=(drug,)))
    processes.append(Process(target=run_frn_model, args=(drug,)))
    processes.append(Process(target=run_selector_model, args=(drug,)))
    processes.append(Process(target=run_extender_frn_model, args=(drug,)))
    processes.append(Process(target=run_extender_selector_model, args=(drug,)))
    #processes.append(Process(target=run_boruta_model_experiment, args=(drug,)))
    processes.append(Process(target=run_extender_robust_model, args=(drug,)))
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    print "done, ", time.time() - start_time
