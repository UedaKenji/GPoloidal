# %%import numpy as np
import matplotlib.pyplot as plt
import zray
import gpoloidal
import gpoloidal.rt1 as rt1
import gpoloidal.plot_utils as plot_utils
import numpy as np

plot_utils.journal_mode()


if __name__ == "__main__":
    # %%
    rt1kernel = rt1.Kernel2D_scatter_rt1()

    # %%
    rt1kernel.set_bound_arange(delta_l=0.02)

    # %%
    rt1kernel.create_inducing_point(r_grid=np.linspace(0.05,1.05,2000), z_grid=np.linspace(-0.7,0.7,2000), length_sq_fuction=rt1.phantom.Length_scale_sq)

    # %%
    rt1kernel.save_point('point_temp',is_plot=True)
    
    print("save point_temp")

    # %%




