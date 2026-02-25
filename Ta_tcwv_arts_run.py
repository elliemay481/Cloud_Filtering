#!/usr/bin/env python

import os
import pyarts
from pyarts.workspace import Workspace, arts_agenda  # type: ignore
import numpy as np  # This example uses numpy
import matplotlib.pyplot as plt

from scipy import constants  # type: ignore
BOLTZMANN_CONST = constants.Boltzmann  # J/K
AVOGADRO_CONST = constants.Avogadro  # 1/mol
H2O_MOLAR_MASS = 0.018015  # kg/mol


def abs_speciesLocate(ws, species) -> int:
    """
    Finds first occurence of a species in abs_species.

    The function only considers the main name, such as "H2O" and "O3",
    and returns the position of the first match in abs_species.

    -1 is returned if no match at all.

    :param ws:      Workspace.
    :param species: Name of species.

    :return: Position of species, -1 if not found.
    """

    for i in range(len(ws.abs_species.value)):
        tag1 = str(ws.abs_species.value[i][0])
        test = tag1.partition("-")[0]
        if test == species:
            return i

    return -1

def vmr2nd(p, t, vmr):
    """
    Converts volume mixing ratio to number density.

    :param p:   Pressure.
    :param t:   Temperature.
    :param vmr: Volume mixing ratio.

    :return: Number density [1/m3].
    """

    return vmr * p / (BOLTZMANN_CONST * t)

fig, ax = plt.subplots(figsize=(9,4))

for h2o_scaling in [0.1, 1, 10]:

    reflectivity=0.5

    # Create a workspace
    ws = pyarts.Workspace()

    # Download ARTS catalogs
    pyarts.cat.download.retrieve()

    ws.iy_main_agendaSet(option="Emission")
    #ws.iy_surface_agendaSet(option="UseSurfaceRtprop")

    @arts_agenda(ws=ws, set_agenda=True)
    def iy_surface_agenda(ws):
        ws.specular_losCalc()
        ws.SurfaceFlatScalarReflectivity(f_reflectivities=[89e9])
        ws.iySurfaceRtpropCalc()

    ws.surface_props_names =  [ "Skin temperature", "Scalar reflectivity 0"]
    ws.surface_props_data = np.array([273.15, reflectivity]).reshape(2, 1, 1)


    #ws.surface_rtprop_agendaSet(option="Blackbody_SurfTFromt_field")
    ws.iy_space_agendaSet(option="CosmicBackground")

    # The path tracing is done step-by-step following a geometric path
    ws.ppath_agendaSet(option="FollowSensorLosPath")
    ws.ppath_step_agendaSet(option="GeometricPath")

    # We might have to compute the Jacobian for the retrieval
    # of relative humidity, so we need to set the agenda for
    # that.
    ws.water_p_eq_agendaSet(option="MK05")

    # The geometry of our planet, and several other properties, are
    # set to those of Earth.
    ws.PlanetSet(option="Earth")

    # Our output unit is Planc brightness temperature
    ws.iy_unit = "PlanckBT"

    # Ignore polarization in this example
    ws.stokes_dim = 1

    # The atmosphere is assumed 1-dimensional
    ws.atmosphere_dim = 1


    # Satellite looking directly nadir, approx. 600 km in altitude
    ws.sensor_pos = [[600e3]]
    ws.sensor_los = [[180]]


    # The dimensions of the problem are defined by a 1-dimensional pressure grid
    ws.p_grid = np.logspace(5.01, -1)
    ws.lat_grid = []
    ws.lon_grid = []


    # The surface is at 0-meters altitude
    ws.z_surface = [[0.0]]


    # AWS channels (groups 2, 3, 4), only centre frequencies
    NF = 6
    ws.f_grid = 10**9 * np.array([
        #89.0,     # 21
        165.5,    # 31
        176.311,  # 32
        178.811,  # 33
        180.311,  # 34
        181.511,  # 35
        182.311,  # 36
        #315.15
        ]) # group 4
    
    # TODO: The proper AWS channels!


    # The atmosphere consists of water, oxygen and nitrogen.
    # We set these to be computed using predefined absorption
    # models.
    ws.abs_speciesSet(species=["H2O-PWR98", "O2-PWR98", "N2-SelfContStandardType"])

    # We have no line-by-line calculations, so we mark the LBL catalog as empty
    ws.abs_lines_per_speciesSetEmpty()

    # We now have all the information required to compute the absorption agenda.
    ws.propmat_clearsky_agendaAuto()


    # Setup atmosphere
    ws.AtmRawRead(basename="planets/Earth/Fascod/subarctic-winter/subarctic-winter")
    #ws.AtmRawRead(basename="planets/Earth/Fascod/tropical/tropical")

    isp_h2o = abs_speciesLocate(ws, "H2O")

    field_h2o = ws.vmr_field_raw.value[isp_h2o]
    new_h2o_vmr = field_h2o.data[:, 0, 0].copy()
    new_h2o_vmr *= h2o_scaling
    field_h2o.data[:, 0, 0] = new_h2o_vmr

    p_field = field_h2o.grids[0]      # pressure grid
    z_field = ws.z_field_raw.value[:,0].flatten()
    t_field = ws.t_field_raw.value[:,0].flatten()

    h2o_vmr = field_h2o[:,0].flatten()
    h2o_nd = vmr2nd(p_field, t_field, h2o_vmr)
    h2o_profile = (H2O_MOLAR_MASS / AVOGADRO_CONST) * h2o_nd
    h2o_column = (H2O_MOLAR_MASS / AVOGADRO_CONST) * np.trapz(
            x=z_field,
            y=h2o_nd
        )
    #ax.plot(h2o_profile, z_field)
    print(h2o_column)
    ws.AtmFieldsCalc()

    # right. i have the column. now to somehow reset it.

    # checkout abs_lookupFascod in easyarts!

    # These calculations do no partial derivatives, so we can turn it off
    ws.jacobianOff()

    # There is no scattering in this example, so we can turn it off
    ws.cloudboxOff()

    # The concept of a sensor does not apply to this example, so we can turn it off
    ws.sensorOff()

    # We check the atmospheric geometry, the atmospheric fields, the cloud box,
    # and the sensor.
    ws.atmgeom_checkedCalc()
    ws.atmfields_checkedCalc()
    ws.cloudbox_checkedCalc()
    ws.sensor_checkedCalc()

    # We perform the calculations
    ws.yCalc()


    ax.plot(np.linspace(0,NF,NF), ws.y.value.value.reshape(NF).T, label=f"TCWV: {np.round(h2o_column, 2)} kg/m2")


ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel(r"$T_{b}$ [K]")
ax.set_xticks(np.linspace(0,NF,NF))
ax.set_xticklabels(ws.f_grid.value.value/1e9)
ax.legend()

fig.tight_layout()

plt.show()
