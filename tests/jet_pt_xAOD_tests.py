# This script should, in theory, allow Pytest to check if a query using ServiceX is actually returning data correctly.
# Requires that ServiceX be running with appropriate port-forward commands on ports 5000 and 9000 for ServixeX and Minio.
# Written by David Liu at the University of Washington, Seattle.
# 17 June 2020

import asyncio
from json import dumps, loads
import queue
import requests
from minio import Minio
import tempfile
import pyarrow.parquet as pq
import pytest
from config import running_backend, default_container
from servicex_test_utils import wait_for_request_done, get_servicex_request_data
import servicex
from func_adl import EventDataset
from func_adl_xAOD import use_exe_servicex
import uproot_methods

# test if we can retrieve the Pts from this particular data set, and that we get back the correct number of lines.
def test_retrieve_simple_jet_pts():
	query = "(call ResultTTree (call Select (call SelectMany (call EventDataset (list 'localds:bogus')) (lambda (list e) (call (attr e 'Jets') 'AntiKt4EMTopoJets'))) (lambda (list j) (/ (call (attr j 'pt')) 1000.0))) (list 'JetPt') 'analysis' 'junk.root')"
	dataset = "mc15_13TeV:mc15_13TeV.361106.PowhegPythia8EvtGen_AZNLOCTEQ6L1_Zee.merge.DAOD_STDM3.e3601_s2576_s2132_r6630_r6264_p2363_tid05630052_00"
	r = servicex.get_data(query, dataset, "http://localhost:5000/servicex")
	assert len(r.index) == 11355980
	
#test the same thing, but using func_adl:
def test_retrieve_simple_jet_pts():
    response = requests.post(f'{"http://localhost:5000/servicex"}/transformation', json={
        "did": "mc15_13TeV:mc15_13TeV.361106.PowhegPythia8EvtGen_AZNLOCTEQ6L1_Zee.merge.DAOD_STDM3.e3601_s2576_s2132_r6630_r6264_p2363_tid05630052_00",
        "selection": "(call ResultTTree (call Select (call SelectMany (call EventDataset (list 'localds:bogus')) (lambda (list e) (call (attr e 'Jets') 'AntiKt4EMTopoJets'))) (lambda (list j) (/ (call (attr j 'pt')) 1000.0))) (list 'JetPt') 'analysis' 'junk.root')",
        "image": default_container,
        "result-destination": "object-store",
        "result-format": "root-file",
        "chunk-size": 1000,
        "workers": 5
    })
    assert response.status_code == 200
    request_id = response.json()["request_id"]
    assert isinstance(request_id, str)

    # Wait for the request to finish
    wait_for_request_done('http://localhost:5000/servicex', request_id)

    # Load the data back.
    pa_table = get_servicex_request_data('http://localhost:5000/servicex', request_id)

    print(pa_table)
    assert len(pa_table) == 11355980
	
#test if we can retrieve the electron four-vectors from this particular data set, and that we get back the correct number of lines.
def test_retrieve_lepton_data():
	query = EventDataset('localds://mc15_13TeV:mc15_13TeV.361106.PowhegPythia8EvtGen_AZNLOCTEQ6L1_Zee.merge.DAOD_STDM3.e3601_s2576_s2132_r6630_r6264_p2363_tid05630052_00')	\
        .Select('lambda e: (e.Electrons("Electrons"), e.Muons("Muons"))') \
        .Select('lambda ls: (ls[0].Select(lambda e: e.pt()), ls[0].Select(lambda e: e.eta()), ls[0].Select(lambda e: e.phi()), ls[0].Select(lambda e: e.e()),ls[1].Select(lambda m: m.pt()), ls[1].Select(lambda m: m.eta()), ls[1].Select(lambda m: m.phi()), ls[1].Select(lambda m: m.e()))') \
        .AsAwkwardArray(('ElePt', 'EleEta', 'ElePhi', 'EleE', 'MuPt', 'MuEta', 'MuPhi', 'MuE')) \
        .value(executor=lambda a: use_exe_servicex(a, endpoint='http://localhost:5000/servicex'))
	four_vector = uproot_methods.TLorentzVectorArray.from_ptetaphi(query[b'ElePt'], query[b'EleEta'],query[b'ElePhi'], query[b'EleE'],)
	four_vector = four_vector[four_vector.counts >= 2]
	dielectrons = four_vector[:, 0] + four_vector[:, 1]
	assert len(dielectrons.mass) == 1502958