import json
import time
import numpy as np
import random
# from sklearn.metrics import confusion_matrix
from azureml.core import Run as AzureRun, ScriptRunConfig, Environment

from . import workspace, log

class Run:
    _INSTANCE = None

    MAX_ACTIVE_CHILDREN = 5

    def __init__(self, remote=None, callback=None):
        self._children = dict()

        self.remote    = remote
        self.callback  = callback
        self._seed     = None

    @staticmethod
    def init():
        if Run._INSTANCE is None:
            # Check if we are in Azure context
            remote = None
            try:
                remote = AzureRun.get_context(allow_offline=False)
            except:
                pass
            Run._INSTANCE = Run(remote=remote)
        return Run._INSTANCE

    @staticmethod
    def is_remote():
        R = Run.init()
        return R.remote is not None

    @staticmethod
    def name(name=None):
        R = Run.init()
        if R.remote is None:
            return None
        else:
            if name is not None:
                R.remote.display_name = name
                log().info(f"Run name set = {name}!")
            return R.remote.display_name

    @staticmethod
    def seed(seed=None):
        R = Run.init()
        if seed is not None:
            if R._seed is not None:
                log().warning(f"Seed already set (= {R._seed}), ignoring new seed {seed}...")
            else:
                R._seed = seed
                random.seed(seed)
                np.random.seed(seed)
                Run.log_metric("Seed", seed)
                log().info(f"Seed set = {R._seed}!")
        return R._seed

    @staticmethod
    def submit_child(
            script,
            arguments=[],
            callback=None,
            name=None,
            tags=None):
        R = Run.init()
        if R.remote is None:
            raise Exception("Local childs not supported yet...")
        
        # Wait until there are less than MAX_ACTIVE_CHILDREN.
        while Run.active_children()>=Run.MAX_ACTIVE_CHILDREN:
            time.sleep(5)

        # There are +1 available spots, create run
        ws = workspace()
        env = R.remote.get_environment()
        ct  = "local"
        src = ScriptRunConfig(source_directory=".", script=script, arguments=arguments, compute_target=ct, environment=env)
        aRc = R.remote.submit_child(src, tags=tags)
        if name is not None: aRc.display_name = name
        log().debug(f"Child run started, name = {name}.")
        Rc = Run(remote=aRc, callback=callback)
        rid = aRc.get_details()["runId"]

        R._children[rid] = Rc
    
    @staticmethod
    def active_children():
        R = Run.init()
        Run.join_children(block=False)
        return len(R._children)

    @staticmethod
    def join_children(block=True):
        R = Run.init()
        joined = 0
        rnd = 0
        while len(R._children)>0:
            rnd += 1
            ncmp = dict()
            for rid,Rc in R._children.items():
                status = Rc.remote.get_status()
                if status in ("Completed","Failed","Canceled"):
                    metrics = Rc.remote.get_metrics()
                    tags    = Rc.remote.get_tags()
                    # Callback
                    log().debug(f"Child joined! RID = {rid}, tags = {tags}")
                    if Rc.callback is not None:
                        Rc.callback(rid, status, metrics=metrics, tags=tags)
                    joined += 1
                else:
                    ncmp[rid] = Rc
            R._children = ncmp
            if not block: break
            log().debug(f"Waiting for children to join ({len(R._children)}), sleeping...")
            time.sleep(5)
        
        return joined

    @staticmethod
    def register_model(model_name, model_path, datasets=[], tags=dict(), properties=dict()):
        R = Run.init()
        if R.remote is not None:
            return R.remote.register_model(
                model_name=model_name,
                model_path=model_path,
                datasets=datasets,
                tags=tags,
                properties=properties
            )
        else:
            raise Exception(f"Error: cannot register model with run - no remote run...")

    @staticmethod
    def log_metric(name, value):
        R = Run.init()
        if R.remote is not None:
            R.remote.log(name, value)
        else:
            log().info(f"Metric logged: {name} = {value}")

    @staticmethod
    def log_row(name, description=None, **kwargs):
        R = Run.init()
        if R.remote is not None:
            R.remote.log_row(name, description=description, **kwargs)
        else:
            log().info(f"Row logged: {name} = {kwargs}")

    @staticmethod
    def set_tags(tags):
        R = Run.init()
        if R.remote is not None:
            R.remote.set_tags(tags)
        else:
            log().info(f"Tags = {tags}")
    
