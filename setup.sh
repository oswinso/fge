# Install et
cd ../
git clone git@github.com:ericyangyu/et.git
cd et
pip install -e . --no-deps

# Install og
cd ../
git clone  git@github.com:oswinso/og.git
cd og
pip install -e . --no-deps

# Install protof16
cd ../
git clone git@github.com:oswinso/protof16.git
cd protof16
pip install -e . --no-deps

# Install jax-f16
cd ../
git clone git@github.com:MIT-REALM/jax-f16.git
cd jax-f16
pip install -e . --no-deps

# Install jax-jumpy
cd ../
git clone git@github.com:oswinso/Jumpy.git
cd Jumpy
pip install -e . --no-deps
cd ../

# Go back to the root dir
cd fge

# Install normal deps.
pip install -U "jax[cuda12]"
pip install ipdb cyclopts colour einops jaxtyping attrs cattrs flowjax loguru optax mujoco-mjx colorcet shapely tqdm wandb pyinstrument playsound3 jax_dataclasses flax diffrax control tensorboard pint flightcondition jax-tqdm "tfp-nightly[jax]" nvtx seaborn "gymnasium==1.1.1" imageio
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install fge
pip install -e . --no-deps