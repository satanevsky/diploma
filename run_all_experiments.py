import sys
import time
from data_keeper import get_data_keeper
from run_experiment import init_common
from run_model_experiment import run_model
from run_selector_model_experiment import run_selector_model
from run_frn_model_experiment import run_frn_model
from run_extender_selector_model_experiment import run_extender_selector_model
from run_extender_frn_model_experiment import run_extender_frn_model


if __name__ == "__main__":
    drug = get_data_keeper().get_possible_second_level_drugs()[int(sys.argv[1])]
    init_common()
    start_time = time.time()
    run_model(drug)
    print "model done, time:", time.time() - start_time
    run_frn_model(drug)
    print "frn_model done, time:", time.time() - start_time
    run_selector_model(drug)
    print "selector_model done, time:", time.time() - start_time
    run_extender_frn_model(drug)
    print "extender_frn done, time:", time.time() - start_time
    run_extender_selector_model(drug)
    print "extender_selector done, time:", time.time() - start_time
    print "done, ", time.time() - start_time
