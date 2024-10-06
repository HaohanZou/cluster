import logging
import os
from pathlib import Path

import pandas as pd
import torch
import torch.multiprocessing as multiprocessing
from dso.utils import log_and_print

from config import get_config
from exp_main import top_main


import os
import random
import time
from datetime import datetime
from typing import Any

import commentjson as json
import numpy as np
import torch
from dotenv import load_dotenv
from dso.config import load_config
from dso.logeval import LogEval
from dso.prior import make_prior
from dso.program import Program
from dso.task import set_task
from dso.utils import log_and_print
from omegaconf import OmegaConf
from torch.multiprocessing import get_logger
import sympy as sym
from dso.program import from_str_tokens, from_tokens



from config import (
    dsoconfig_factory,
    nesymres_dataset_config_factory,
    nesymres_function_set_factory,
    nesymres_train_config_factory,
)
from datasets import generate_train_and_val_functions
from models.controllers import LSTMController, TransformerController
from models.dso_controller import DsoController
from models.transformers2 import (
    TransformerTreeController,
    TransformerTreeEncoderController,
)
from utils.train import (
    gp_at_test,
    optomize_at_test,
    sample_nesymres_at_test,
    train_controller,
)
from utils.train_ce import train_encoder_ce_controller

logger = get_logger()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Using {} device".format(DEVICE))

gradient_clip = 1.0




conf = get_config()

conf.exp.seed_runs = 1
conf.exp.n_cores_task = 1  # 7 if GPU memory is at least 24GB, else tune to be smaller
conf.exp.seed_start = 7
conf.exp.baselines = ["Transformer"]
# User must specify the benchmark to run:
conf.exp.benchmark = "fn_d_all_m"  # Possible values ["fn_d_2", "fn_d_5", "l_cd_12", ""fn_d_all"]

Path("./logs").mkdir(parents=True, exist_ok=True)

benchmark_df = pd.read_csv(conf.exp.benchmark_path, index_col=0, encoding="ISO-8859-1")
df = benchmark_df[benchmark_df.index.str.contains(conf.exp.benchmark)]
datasets = df.index.to_list()

file_name = os.path.basename(os.path.realpath(__file__)).split(".py")[0]
path_run_name = "all_{}-{}_01".format(file_name, conf.exp.benchmark)


def create_our_logger(path_run_name):
    logger = multiprocessing.get_logger()
    formatter = logging.Formatter("%(processName)s| %(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s")
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler("./logs/{}_log.txt".format(path_run_name))
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    logger.info("STARTING NEW RUN ==========")
    logger.info(f"SEE LOG AT : ./logs/{path_run_name}_log.txt")
    return logger



logger = create_our_logger(path_run_name)
logger.info(f"See log at : ./logs/{path_run_name}_log.txt")
data_samples_to_use = int(float(df["train_spec"][0].split(",")[-1].split("]")[0]) * conf.exp.dataset_size_multiplier)

def perform_run(tuple_in):
    seed, dataset, baseline = tuple_in
    logger.info(
        f"[BASELINE_TESTING NOW] dataset={dataset} \t| baseline={baseline} \t| seed={seed} \t| data_samples={data_samples_to_use} \t| noise={conf.exp.noise}"
    )
    
    # try:
    if baseline == "NGGP":
        result = top_main(
            test_dataset=dataset,
            seed=seed,
            training_equations=200000,
            training_epochs=100,
            batch_outer_datasets=24,
            batch_inner_equations=100,
            pre_train=False,
            load_pre_trained_path="",
            priority_queue_training=conf.exp.priority_queue_training,
            gp_meld=conf.gp_meld.run_gp_meld,
            model="dso",
            train_path="",
            test=conf.exp.run_pool_programs_test,
            risk_seeking_pg_train=True,
            save_true_log_likelihood=conf.exp.save_true_log_likelihood,
            p_crossover=conf.gp_meld.p_crossover,
            p_mutate=conf.gp_meld.p_mutate,
            tournament_size=conf.gp_meld.tournament_size,
            generations=conf.gp_meld.generations,
            function_set=conf.exp.function_set,
            learning_rate=conf.exp.learning_rate,
            test_sample_multiplier=conf.exp.test_sample_multiplier,
            n_samples=conf.exp.n_samples,
            dataset_size_multiplier=conf.exp.dataset_size_multiplier,
            noise=conf.exp.noise,
            )
    else:
    
        result = top_main(
                test_dataset=dataset,
                seed=seed,
                training_equations=200000,
                training_epochs=100,
                batch_outer_datasets=24,
                batch_inner_equations=100,
                pre_train=True,
                skip_pre_training=True,
                load_pre_trained_path="",
                priority_queue_training=conf.exp.priority_queue_training,
                gp_meld=conf.gp_meld.run_gp_meld,
                model="TransformerTreeEncoderController",
                train_path="",
                test=conf.exp.run_pool_programs_test,
                risk_seeking_pg_train=True,
                save_true_log_likelihood=conf.exp.save_true_log_likelihood,
                p_crossover=conf.gp_meld.p_crossover,
                p_mutate=conf.gp_meld.p_mutate,
                tournament_size=conf.gp_meld.tournament_size,
                generations=conf.gp_meld.generations,
                function_set=conf.exp.function_set,
                learning_rate=conf.exp.learning_rate,
                test_sample_multiplier=conf.exp.test_sample_multiplier,
                n_samples=conf.exp.n_samples,
                dataset_size_multiplier=conf.exp.dataset_size_multiplier,
                noise=conf.exp.noise,
            )
        
    result["baseline"] = baseline  # pyright: ignore
    result["run_seed"] = seed  # pyright: ignore
    result["dataset"] = dataset  # pyright: ignore
    log_and_print(f"[TEST RESULT] {result}")  # pyright: ignore
    return result  # pyright: ignore


def main(dataset, n_cores_task=conf.exp.n_cores_task):

    task_inputs = []
    for seed in range(conf.exp.seed_start, conf.exp.seed_start + conf.exp.seed_runs):
        for baseline in conf.exp.baselines:
            task_inputs.append((seed, dataset, baseline))

    if n_cores_task is None:
        n_cores_task = multiprocessing.cpu_count()
    if n_cores_task >= 2:
        pool_outer = multiprocessing.Pool(n_cores_task)
        for i, result in enumerate(pool_outer.imap(perform_run, task_inputs)):
            log_and_print(
                "INFO: Completed run {} of {} in {:.0f} s | LATEST TEST_RESULT {}".format(
                    i + 1, len(task_inputs), result["t"], result
                )
            )
    else:
        for i, task_input in enumerate(task_inputs):
            result = perform_run(task_input)
            log_and_print(
                "INFO: Completed run {} of {} in {:.0f} s | LATEST TEST_RESULT {}".format(
                    i + 1, len(task_inputs), result["t"], result
                )
            )


# Functhions from exp_main.py
            
def top_main(
    test_dataset="fey-sim-53",
    seed=0,
    training_equations=200000,
    training_epochs=100,
    batch_outer_datasets=24,
    batch_inner_equations=100,
    pre_train=True,  # Pre-train model type
    skip_pre_training=False,
    load_pre_trained_path="/home/sam/code/scaling_discovery/log/Nguyen-10_2022-05-05-203406/",
    priority_queue_training=True,
    gp_meld=True,
    model="TransformerTreeEncoderController",
    train_path="",
    test=False,
    risk_seeking_pg_train=True,
    test_sample_multiplier=1,
    data_gen_max_len=20,  # 256,
    data_gen_max_ops=5,  # 41,
    data_gen_equal_prob_independent_vars=False,
    data_gen_remap_independent_vars_to_monotic=False,
    data_gen_force_all_independent_present=False,
    data_gen_operators=None,
    data_gen_lower_nbs_ops=3,
    data_gen_create_eqs_with_constants=False,
    use_latest_DSRNG_hyperparameters=True,
    save_true_log_likelihood=False,
    p_crossover=None,
    p_mutate=None,
    tournament_size=None,
    generations=None,
    function_set=None,
    learning_rate=None,
    rl_weight=1.0,
    epsilon=None,
    n_samples=None,
    dataset_size_multiplier=1.0,
    noise=0.0,
):  
    seed_all(seed)
    load_dotenv()
    CPU_COUNT_DIV = int(os.getenv("CPU_COUNT_DIV")) if os.getenv("CPU_COUNT_DIV") else 1  # pyright: ignore
    log_and_print(
        f"[RUN SETTINGS]: test_dataset={test_dataset} training_equations={training_equations} "
        f"training_epochs={training_epochs} batch_outer_datasets={batch_outer_datasets} pre_train={pre_train} "
        f"load_pre_trained_path={load_pre_trained_path} priority_queue_training={priority_queue_training} "
        f"gp_meld={gp_meld} model={model} train_path={train_path} risk_seeking_pg_train={risk_seeking_pg_train}"
    )
    
    dsoconfig = dsoconfig_factory()
    nesymres_dataset_config = nesymres_dataset_config_factory()
    nesymres_train_config: Any = nesymres_train_config_factory()
    nesymres_function_set = nesymres_function_set_factory()

    if risk_seeking_pg_train:
        batch_outer_datasets = 5
    else:
        batch_outer_datasets = os.cpu_count() // CPU_COUNT_DIV  # pyright: ignore
    if load_pre_trained_path:
        if not (load_pre_trained_path[-1] == "/"):
            raise ValueError("pre-trained path input incorrect - please fix ... : {}".format(load_pre_trained_path))
        with open(load_pre_trained_path + "config.json") as fh:
            load_pre_trained_config = json.load(fh)
    else:
        load_pre_trained_config = None

    # Determine library of functions, i.e. the function set name
    if function_set is not None:
        dsoconfig["task"]["function_set"] = function_set
    dsoconfig["task"]["dataset"] = test_dataset
    config = load_config(dsoconfig)
    config["controller"]["pqt"] = priority_queue_training
    config["gp_meld"]["run_gp_meld"] = gp_meld
    config["model"] = model
    log_and_print("Running model : {}".format(model))
    Program.clear_cache()
    complexity = config["training"]["complexity"]
    Program.set_complexity(complexity)
    

    # Set the constant optimizer
    const_optimizer = config["training"]["const_optimizer"]
    const_params = config["training"]["const_params"]
    const_params = const_params if const_params is not None else {}
    Program.set_const_optimizer(const_optimizer, **const_params)

    pool = None
    # Set the Task for the parent process
    set_task(config["task"])

    
    test_task: Any = Program.task
    function_names = test_task.library.names
    function_set_name = test_task.function_set_name
    logger.info("Function set: {}".format(function_names))
    # seed_all(seed)
    '''
    if load_pre_trained_config:
        print(True)
        load_pre_trained_function_names = load_pre_trained_config["function_names_str"]
        cond_list = [i[0] == "x" for i in load_pre_trained_function_names.split(",") if i not in function_names]
        only_missing_vars = all(cond_list) and (len(cond_list) >= 1)
        if load_pre_trained_function_names != ",".join(function_names) and not only_missing_vars:
            log_and_print(
                "Incorrect function set with pre-trained model ... aborting \t| trained with: {} \t != task: {}".format(
                    load_pre_trained_function_names, ",".join(function_names)
                )
            )
            raise ValueError("Incorrect function set with pre-trained model ... aborting")
        elif only_missing_vars:
            config["task"]["n_input_var"] = len([j for j in load_pre_trained_function_names.split(",") if j[0] == "x"])
            set_task(config["task"])
            test_task = Program.task
            function_names = test_task.library.names
            function_set_name = test_task.function_set_name
            logger.info("Function set: {}".format(function_names))
        load_pre_trained_model = load_pre_trained_config["model"]
        if load_pre_trained_model != model:
            log_and_print(
                "Incorrect model trying to load ... aborting \t| {} != {}".format(load_pre_trained_model, model)
            )
    '''
    set_task(config["task"])
    number_of_independent_vars = len(["x_" + fn[1] for fn in function_names if "x" == fn[0]])

    # Convert function_names to generator compatible ones
    '''
    nesymres_dataset_config["variables"] = ["x_" + fn[1] for fn in function_names if "x" == fn[0]]
    number_of_independent_vars = len(["x_" + fn[1] for fn in function_names if "x" == fn[0]])
    nesymres_dataset_config["max_len"] = data_gen_max_len
    nesymres_dataset_config["max_ops"] = data_gen_max_ops
    nesymres_dataset_config["equal_prob_independent_vars"] = data_gen_equal_prob_independent_vars
    nesymres_dataset_config["remap_independent_vars_to_monotic"] = data_gen_remap_independent_vars_to_monotic
    nesymres_dataset_config["force_all_independent_present"] = data_gen_force_all_independent_present
    if data_gen_operators:
        nesymres_dataset_config["operators"] = data_gen_operators
    nesymres_dataset_config["max_independent_vars"] = number_of_independent_vars
    nesymres_dataset_config["lower_nbs_ops"] = data_gen_lower_nbs_ops
    nesymres_dataset_config["create_eqs_with_constants"] = data_gen_create_eqs_with_constants

    train_sampling_type = list(test_task.train_spec["x1"].keys())[0]
    nesymres_train_config["num_of_workers"] = os.cpu_count() // CPU_COUNT_DIV  # pyright: ignore
    nesymres_train_config["dataset_train"]["type_of_sampling_points"] = (
        "constant" if train_sampling_type == "U" else "logarithm"
    )
    nesymres_train_config["dataset_train"]["fun_support"]["min"] = test_task.train_spec["x1"][train_sampling_type][0]
    nesymres_train_config["dataset_train"]["fun_support"]["max"] = test_task.train_spec["x1"][train_sampling_type][1]
    nesymres_train_config["dataset_train"]["max_number_of_points"] = test_task.train_spec["x1"][train_sampling_type][2]
    nesymres_train_config["dataset_train"]["padding_token"] = test_task.library.L
    nesymres_train_config["dataset_val"]["type_of_sampling_points"] = (
        "constant" if train_sampling_type == "U" else "logarithm"
    )
    nesymres_train_config["dataset_val"]["fun_support"]["min"] = test_task.train_spec["x1"][train_sampling_type][0]
    nesymres_train_config["dataset_val"]["fun_support"]["max"] = test_task.train_spec["x1"][train_sampling_type][1]
    nesymres_train_config["dataset_val"]["max_number_of_points"] = test_task.train_spec["x1"][train_sampling_type][2]

    test_sampling_type = list(test_task.test_spec["x1"].keys())[0]
    nesymres_train_config["dataset_test"]["type_of_sampling_points"] = (
        "constant" if test_sampling_type == "U" else "logarithm"
    )
    nesymres_train_config["dataset_test"]["fun_support"]["min"] = test_task.test_spec["x1"][test_sampling_type][0]
    nesymres_train_config["dataset_test"]["fun_support"]["max"] = test_task.test_spec["x1"][test_sampling_type][1]
    nesymres_train_config["dataset_test"]["max_number_of_points"] = test_task.test_spec["x1"][test_sampling_type][2]
    nesymres_train_config["batch_size"] = batch_outer_datasets

    
    nesymres_train_config["architecture"]["length_eq"] = config["prior"]["length"]["max_"]
    '''

    nesymres_train_config["architecture"]["dim_input"] = 64

    # DSO setup
    # Save starting seed
    config["experiment"]["starting_seed"] = config["experiment"]["seed"]
    # Set timestamp once to be used by all workers
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    config["experiment"]["timestamp"] = timestamp
    config["training"]["batch_size"] = batch_inner_equations
    if n_samples is not None:
        config["training"]["n_samples"] = n_samples
    if epsilon is None:
        config["training"]["epsilon"] = None  # Turn off RSG PG, want to pre-train with VPG
        config["training"]["baseline"] = "ewma_R_e"
    else:
        config["training"]["epsilon"] = epsilon
        config["training"]["baseline"] = "R_e"

    # Save complete configuration file
    output_file = make_output_file(config, seed)
    controller_saved_path = "/".join(output_file.split("/")[:-1]) + "/" + "controller.pt"  # pyright: ignore
    config["function_names_str"] = ",".join(function_names)  # Order very important here

    # Prepare training parameters
    if learning_rate is None:
        config["controller"]["learning_rate"] = 0.001  # 0.0001 previous
    else:
        config["controller"]["learning_rate"] = learning_rate  # 0.0001 previous
    config["controller"]["entropy_weight"] = 0.01 # 0.003 previous # 0.005 previous
    config["controller"]["entropy_gamma"] = 0.8 # 0.7 previous  # 0.8 previous
    # config['controller']['entropy_gamma'] = 0.8 # 0.8 previous
    config["state_manager"]["embedding"] = False  # True previous
    config["state_manager"]["embedding_size"] = 32  # 32 previous  # 16 also good, 32 best
    config["controller"]["rl_weight"] = rl_weight  # Default 1.0
    
    
    prior = make_prior(test_task.library, config["prior"])
    if model == "dso":
        controller: Any = DsoController(
            prior,
            test_task.library,
            test_task,
            config_state_manager=config["state_manager"],
            # encoder_input_dim=number_of_independent_vars + 1,
            **config["controller"],
        ).to(DEVICE)
    '''
    elif model == "mlp_encoder_lstm_decoder":
        controller = LSTMController(
            prior,
            test_task.library,
            test_task,
            config_state_manager=config["state_manager"],
            # encoder_input_dim=number_of_independent_vars + 1,
            **config["controller"],
        ).to(DEVICE)
    elif model == "set_transformer_transformer_decoder":
        # config['training']['learning_rate'] = nesymres_train_config['architecture']['lr']
        controller = TransformerController(
            prior,
            test_task.library,
            test_task,
            nesymres_train_config["architecture"],
            config_state_manager=config["state_manager"],
            encoder_input_dim=number_of_independent_vars + 1,
            **config["controller"],
        ).to(DEVICE)
    elif model == "TransformerTreeController":
        config["state_manager"]["embedding"] = True
        config["state_manager"]["embedding_size"] = 16  # 16 also good - 64 was the best setting
        config["controller"]["num_units"] = 32
        controller = TransformerTreeController(
            prior,
            test_task.library,
            test_task,
            config_state_manager=config["state_manager"],
            encoder_input_dim=number_of_independent_vars + 1,
            **config["controller"],
        ).to(DEVICE)
        '''
    if model == "TransformerTreeEncoderController":
        config["state_manager"]["embedding"] = True
        config["state_manager"]["embedding_size"] = 128  # 64 also good
        config["controller"]["num_units"] = 128
        controller = TransformerTreeEncoderController(
            prior,
            test_task.library,
            test_task,
            nesymres_train_config["architecture"],
            config_state_manager=config["state_manager"],
            encoder_input_dim= config["controller"]["num_units"],
            **config["controller"],
            vocab_size=31 
        ).to(DEVICE)
    '''
    elif model == "nesymres":
        from models.adapted_nesrymres import get_nesrymres_model

        controller = get_nesrymres_model()
    elif model == "gp":
        from dso.baselines.gpsr import GP
        from dso.task.regression.dataset import BenchmarkDataset

        dataset = BenchmarkDataset(name=test_dataset)
        library_names = test_task.library.names
        library_names = [i for i in library_names if not "x" == i[0]]
        dataset.function_set = library_names
        controller = GP(dataset, verbose=True, pareto_front=True, seed=seed)
    '''
    if model != "gp":
        log_and_print(
            f"{model} parameters: {sum(p.numel() for p in controller.parameters())} \t | "  # pyright: ignore
            f"trainable : {sum(p.numel() for p in controller.parameters() if p.requires_grad)}"  # pyright: ignore
        )
        torch.save(controller.state_dict(), controller_saved_path)  # pyright: ignore
    config["nesymres_train_config"] = OmegaConf.to_container(nesymres_train_config, resolve=True)
    save_config(output_file, config)
    if config["gp_meld"].pop("run_gp_meld", False):
        config["gp_meld"]["train_n"] = test_sample_multiplier * 50
        if p_crossover is not None:
            config["gp_meld"]["p_crossover"] = p_crossover
        if p_mutate is not None:
            config["gp_meld"]["p_mutate"] = p_mutate
        if tournament_size is not None:
            config["gp_meld"]["tournament_size"] = tournament_size
        if generations is not None:
            config["gp_meld"]["generations"] = generations
        log_and_print("GP CONFIG : {}".format(config["gp_meld"]))
        from dso.gp.gp_controller import GPController

        del config["gp_meld"]["verbose"]
        gp_controller = GPController(prior, pool, **config["gp_meld"], seed=seed)
        # gp_controller.return_gp_obs = False
    else:
        gp_controller = None


    logger.info("config: {}".format(config))

    '''
    if pre_train and not load_pre_trained_path and not skip_pre_training:
        # Generate lots of equations with that function set and save to data folder
        print(True)
        dltrain, dlval, dltest, train_path = generate_train_and_val_functions(
            function_names,
            function_set_name,
            nesymres_train_config,
            nesymres_dataset_config,
            training_equations=training_equations,
            train_path=train_path,
            train_global_seed=99,
        )
        config["pre_train_dataset_path"] = str(train_path)
        log_and_print(f"Dataset train path: {train_path}")
        save_config(output_file, config)
        if risk_seeking_pg_train:
            controller = train_controller(
                dltrain,
                dlval,
                dltest,
                controller,  # pyright: ignore
                pool,
                gp_controller,
                output_file,
                pre_train,
                training_epochs=training_epochs,
                batch_outer_datasets=batch_outer_datasets,
                batch_inner_equations=batch_inner_equations,
                load_pre_trained_path=load_pre_trained_path,
                config=config,
                controller_saved_path=controller_saved_path,
                **config["training"],
            )
        else:
            # pylint: disable-next=assignment-from-no-return
            controller = train_encoder_ce_controller(
                dltrain,
                dlval,
                dltest,
                controller,  # pyright: ignore
                pool,
                gp_controller,
                output_file,
                pre_train,
                training_epochs=training_epochs,
                batch_outer_datasets=batch_outer_datasets,
                batch_inner_equations=batch_inner_equations,
                load_pre_trained_path=load_pre_trained_path,
                config=config,
                controller_saved_path=controller_saved_path,
                **config["training"],
            )
    
    elif load_pre_trained_path:
        controller.load_state_dict(torch.load(load_pre_trained_path + "controller.pt"))  # pyright: ignore
    
    '''
   
    # if model == "TransformerTreeEncoderController":
        # for param in controller.model.data_encoder.parameters():  # pyright: ignore
            # param.requires_grad = False
    Program.clear_cache()
    set_task(config["task"])

    test_start = time.time()
    result = {"seed": seed}  # Seed listed first
    controller.save_true_log_likelihood = save_true_log_likelihood  # pyright: ignore
    controller.true_eq = []  # pyright: ignore
    if use_latest_DSRNG_hyperparameters:
        config["training"]["batch_size"] = 500 * test_sample_multiplier
        config["training"]["epsilon"] = 0.8
        config["training"]["baseline"] = "ewma_R"
        # config["training"]["baseline"] = "R_e"
    else:
        config["training"]["batch_size"] = 1000
        config["training"]["epsilon"] = 0.05
        config["training"]["baseline"] = "R_e"
        controller.learning_rate = 0.0005  # pyright: ignore
        controller.entropy_weight = 0.005  # pyright: ignore
    '''
    if save_true_log_likelihood:

        from dso.task.regression.dataset import BenchmarkDataset

        dataset = BenchmarkDataset(name=test_dataset)
        f_str = dataset.function_expression

        from dso.program import from_tokens
        from nesymres.architectures.data_utils import (
            eq_sympy_prefix_to_token_library,
            replace_with_div,
            replace_with_neg_with_sub,
        )

        # eq_remove_constants,
        from nesymres.dataset.generator import Generator
        from sympy.parsing.sympy_parser import parse_expr

        expr = parse_expr(f_str, evaluate=True)
        prefix = Generator.sympy_to_prefix(expr)
        eq = eq_sympy_prefix_to_token_library(prefix)
        eq = replace_with_div(eq)
        eq = replace_with_neg_with_sub(eq)
        eq = [f"{int(e):.1f}" if e.isdigit() else e for e in eq]
        eq = "add,mul,add,div,mul,x2,x2,add,x2,x2,x2,x1,sub,x1,x1".split(",")
        true_a = test_task.library.actionize(eq)
        p = from_tokens(true_a)
        log_and_print("True equation")
        p.print_stats()
    else:
        true_a = None
    '''   
    true_a = None

    if model == "nesymres":
        result.update(
            sample_nesymres_at_test(
                controller,  # pyright: ignore
                pool,
                gp_controller,
                output_file,
                pre_train,
                config,
                test_task,
                controller_saved_path=controller_saved_path,
                **config["training"],
                save_true_log_likelihood=save_true_log_likelihood,
                true_action=true_a,
            )
        )
    elif model == "gp":
        result.update(
            gp_at_test(  # pyright: ignore
                controller,  # pyright: ignore
                pool,
                gp_controller,
                output_file,
                pre_train,
                config,
                test_task,
                controller_saved_path=controller_saved_path,
                **config["training"],
                save_true_log_likelihood=save_true_log_likelihood,
                true_action=true_a,
            )
        )
    else:
        result.update(
            optomize_at_test(
                controller,  # pyright: ignore
                pool,
                gp_controller,
                output_file,
                pre_train,
                config,
                controller_saved_path=controller_saved_path,
                **config["training"],
                save_true_log_likelihood=save_true_log_likelihood,
                true_action=true_a,
            )
        )
    result["t"] = time.time() - test_start  # pyright: ignore
    

    save_path = config["experiment"]["save_path"]
    summary_path = os.path.join(save_path, "summary.csv")

    log_and_print("== TRAINING SEED {} END ==============".format(config["experiment"]["seed"]))

    # Evaluate the log files
    log_and_print("\n== POST-PROCESS START =================")
    log = LogEval(config_path=os.path.dirname(summary_path))
    log.analyze_log(
        show_count=config["postprocess"]["show_count"],
        show_hof=config["training"]["hof"] is not None and config["training"]["hof"] > 0,
        show_pf=config["training"]["save_pareto_front"],
        save_plots=config["postprocess"]["save_plots"],
    )
    log_and_print("== POST-PROCESS END ===================")
    return result


def save_config(output_file, config):
    # Save the config file
    if output_file is not None:
        path = os.path.join(config["experiment"]["save_path"], "config.json")
        # With run.py, config.json may already exist. To avoid race
        # conditions, only record the starting seed. Use a backup seed
        # in case this worker's seed differs.
        backup_seed = config["experiment"]["seed"]
        if not os.path.exists(path):
            if "starting_seed" in config["experiment"]:
                config["experiment"]["seed"] = config["experiment"]["starting_seed"]
                del config["experiment"]["starting_seed"]
            with open(path, "w") as f:
                json.dump(config, f, indent=3)
        config["experiment"]["seed"] = backup_seed


def make_output_file(config, seed):
    """Generates an output filename"""

    # If logdir is not provided (e.g. for pytest), results are not saved
    if config["experiment"].get("logdir") is None:
        logger.info("WARNING: logdir not provided. Results will not be saved to file.")
        return None

    # When using run.py, timestamp is already generated
    timestamp = config["experiment"].get("timestamp")
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        config["experiment"]["timestamp"] = timestamp

    # Generate save path
    task_name = Program.task.name  # pyright: ignore
    save_path = os.path.join(config["experiment"]["logdir"], "_".join([task_name, timestamp, str(seed)]))
    config["experiment"]["task_name"] = task_name
    config["experiment"]["save_path"] = save_path
    os.makedirs(save_path, exist_ok=True)

    seed = config["experiment"]["seed"]
    output_file = os.path.join(save_path, "dso_{}_{}.csv".format(task_name, seed))

    return output_file


def seed_all(seed=None):
    """
    Set the torch, numpy, and random module seeds based on the seed
    specified in config. If there is no seed or it is None, a time-based
    seed is used instead and is written to config.
    """
    # Default uses current time in milliseconds, modulo 1e9
    if seed is None:
        seed = round(time() * 1000) % int(1e9)  # pyright: ignore  # pylint: disable=not-callable

    # Set the seeds using the shifted seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if output_file is not None:
        path = os.path.join(config["experiment"]["save_path"], "config.json")
        # With run.py, config.json may already exist. To avoid race
        # conditions, only record the starting seed. Use a backup seed
        # in case this worker's seed differs.
        backup_seed = config["experiment"]["seed"]
        if not os.path.exists(path):
            if "starting_seed" in config["experiment"]:
                config["experiment"]["seed"] = config["experiment"]["starting_seed"]
                del config["experiment"]["starting_seed"]
            with open(path, "w") as f:
                json.dump(config, f, indent=3)
        config["experiment"]["seed"] = backup_seed


def make_output_file(config, seed):
    """Generates an output filename"""

    # If logdir is not provided (e.g. for pytest), results are not saved
    if config["experiment"].get("logdir") is None:
        logger.info("WARNING: logdir not provided. Results will not be saved to file.")
        return None

    # When using run.py, timestamp is already generated
    timestamp = config["experiment"].get("timestamp")
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        config["experiment"]["timestamp"] = timestamp

    # Generate save path
    task_name = Program.task.name  # pyright: ignore
    save_path = os.path.join(config["experiment"]["logdir"], "_".join([task_name, timestamp, str(seed)]))
    config["experiment"]["task_name"] = task_name
    config["experiment"]["save_path"] = save_path
    os.makedirs(save_path, exist_ok=True)

    seed = config["experiment"]["seed"]
    output_file = os.path.join(save_path, "dso_{}_{}.csv".format(task_name, seed))

    return output_file


def seed_all(seed=None):
    """
    Set the torch, numpy, and random module seeds based on the seed
    specified in config. If there is no seed or it is None, a time-based
    seed is used instead and is written to config.
    """
    # Default uses current time in milliseconds, modulo 1e9
    if seed is None:
        seed = round(time() * 1000) % int(1e9)  # pyright: ignore  # pylint: disable=not-callable

    # Set the seeds using the shifted seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

'''
class Node(object):
    """Basic tree class supporting printing"""

    def __init__(self, val):
        self.val = val
        self.children = []

    def __repr__(self):
        children_repr = ",".join(repr(child) for child in self.children)
        if len(self.children) == 0:
            return self.val  # Avoids unnecessary parentheses
        return "{}({})".format(self.val, children_repr)


def parse_expression(expr):
    """Converts a SymPy expression into a tree structure"""
    if expr.is_Add:
        # Handle addition and subtraction as separate operations
        terms = list(expr.args)
        root = None
        while terms:
            left = terms.pop(0)
            if terms and terms[0].is_negative:
                right = terms.pop(0)
                node = Node("sub")
                node.children.append(parse_expression(left))
                node.children.append(parse_expression(-right))
            else:
                if root is None:
                    root = parse_expression(left)
                if terms:
                    next_term = terms.pop(0)
                    if next_term.is_negative:
                        node = Node("sub")
                        node.children.append(root)
                        node.children.append(parse_expression(-next_term))
                    else:
                        node = Node("add")
                        node.children.append(root)
                        node.children.append(parse_expression(next_term))
                    root = node
        return root

    elif expr.is_Mul:
        # Handle multiplication
        node = Node("mul")
        for arg in expr.args:
            node.children.append(parse_expression(arg))
        return node

    elif expr.is_Pow:
        # Handle exponentiation with specific handling for squared terms
        base, exp = expr.args
        if exp == 2:
            node = Node("n2")
        elif exp == 3:
            node = Node("n3")
        elif exp == 4:
            node = Node("n4")
        else:
            node = Node("pow")
            node.children.append(parse_expression(base))
            node.children.append(parse_expression(exp))
        if exp in {2, 3, 4}:
            node.children.append(parse_expression(base))
        return node

    elif expr.is_Symbol or expr.is_Number:
        # Handle symbols and numbers
        return Node(str(expr))

    else:
        # Handle other cases (e.g., negative numbers)
        if expr.is_negative:
            node = Node("sub")
            node.children.append(Node("0"))
            node.children.append(parse_expression(-expr))
            return node

        raise ValueError(f"Unsupported operation: {expr}")



def preorder_traversal(tree):
    """Generate pre-order traversal string of the tree"""
    def traverse(node):
        if len(node.children) == 0:
            return [node.val]
        result = [node.val]
        for child in node.children:
            result.extend(traverse(child))
        return result

    traversal = traverse(tree)
    return ",".join(traversal)

'''

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    from dso.config import load_config
    from dso.task import set_task

    from config import (
        dsoconfig_factory,
        nesymres_dataset_config_factory,
        nesymres_function_set_factory,
        nesymres_train_config_factory,
    )
    
    dsoconfig = dsoconfig_factory()
    log_and_print(df.to_string())
    for dataset, row in df.iterrows():
    
        covars = row["variables"]

        nesymres_dataset_config = nesymres_dataset_config_factory()
        nesymres_train_config = nesymres_train_config_factory()
        nesymres_function_set = nesymres_function_set_factory()
        dsoconfig["task"]["dataset"] = dataset
        config = load_config(dsoconfig)
        set_task(config["task"])
        try:
            main(dataset)
        except FileNotFoundError as e:
            # pylint: disable-next=raise-missing-from
            if 'nesymres_pre_train' in str(e):
                raise FileNotFoundError(
                    f"Please download the baseline pre-trained models for NeuralSymbolicRegressionThatScales from https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales and put them into the folder `models/nesymres_pre_train`. No pre-trained model of {e.filename} in folder './models/pre_train/' for covars={covars}. "
                )
            else:                
                raise FileNotFoundError(
                    f"No pre-trained model of {e.filename} in folder './models/pre_train/' for covars={covars}. "
                )
    logger.info("Fin.")
    
