# This script checks if a query using ServiceX is actually returning data correctly. This is the Uproot version.
# Requires that ServiceX be running with appropriate port-forward commands on ports 5000 and 9000 for ServiceX and Minio.
# Written by David Liu at the University of Washington, Seattle.
# 6 July 2020

from func_adl_servicex import ServiceXSourceUpROOT
import pandas as pd
import logging


def test_retrieve_simple_jet_pts_uproot(caplog):
    'test if we can retrieve the Pts from this particular data set, and that we get back the correct number of lines.'
    caplog.set_level(logging.DEBUG)

    dataset_uproot = "data15_13TeV:data15_13TeV.00282784.physics_Main.deriv.DAOD_PHYSLITE.r9264_p3083_p4165_tid21568807_00"

    ds = ServiceXSourceUpROOT(dataset_uproot, "CollectionTree")
    data = ds.Select("lambda e: {'JetPT': e['AnalysisJetsAuxDyn.pt']}") \
        .AsAwkwardArray() \
        .value()

    columnar_data = pd.read_parquet(data)

    assert len(columnar_data.JetPT) == 4046640

# # test if lambda capture is functional for uproot backend
# # Test not implemented yet, so commenting out
# def test_lambda_capture():
#     dataset_uproot = "data15_13TeV:data15_13TeV.00282784.physics_Main.deriv.DAOD_PHYSLITE.r9264_p3083_p4165_tid21568807_00"

#     ds = ServiceXSourceUpROOT(dataset_uproot, "CollectionTree")
#     data = ds.Select("lambda e: e.Select(lambda jet: {'JetPt': jet.pt})")
