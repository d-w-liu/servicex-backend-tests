# This script stress tests ServiceX.
# Requires that ServiceX be running with appropriate port-forward commands on ports 5000 and 9000 for ServiceX and Minio.
# Written by David Liu at the University of Washington, Seattle.
# 20 July 2020

import asyncio
import servicex
from servicex import ServiceXDataset
from servicex.minio_adaptor import MinioAdaptor
from servicex.servicex_adaptor import ServiceXAdaptor
from func_adl_servicex import ServiceXSourceXAOD
import uproot_methods
from numpy import genfromtxt
import numpy as np
import time
import pytest

# pytestmark = run_stress_tests # stress tests take upwards of 40 minutes to complete, so we don't do them unless we want to.

# this function tests the ability of ServiceX to pull one hundred column of data from a 10 TB dataset
@pytest.mark.stress
def test_servicex_10TB_capacity():
    dataset = ServiceXDataset('data16_13TeV:data16_13TeV.AllYear.physics_Main.PhysCont.DAOD_TOPQ5.grp16_v01_p4173', max_workers = 400) # define which dataset we want

    # here, we build the relevant query and send it to ServiceX. Refer to https://github.com/iris-hep/func_adl/blob/master/documentation.md for how to build the query.
    query = ServiceXDatasetSource(dataset) \
        .Select('lambda e: (e.Jets("AntiKt4EMTopoJets"), \
                            e.Tracks("AntiKt4PV0TrackJets"), \
                            e.BTopo("BTagging_AntiKT4EMTopo.index"), \
                            e.BTrack("BTagging_AntiKt4Track"), \
                            e.Clusters("CaloCalTopoClusters"), \
                            e.MuonTracks("CombinedMuonTrackParticles"), \
                            e.Electrons("Electrons"), \
                            e.GSFTrack("GSFTrackParticles"), \
                            e.Muons("Muons"), \
                            e.Photons("Photons"))') \
        .Select('lambda ls: (ls[0].Select(lambda jet: jet.e()), \
                             ls[0].Select(lambda jet: jet.eta()), \
                             ls[0].Select(lambda jet: jet.index()), \
                             ls[0].Select(lambda jet: jet.m()), \
                             ls[0].Select(lambda jet: jet.phi()), \
                             ls[0].Select(lambda jet: jet.pt().Where(lambda jet: jet.pt()/1000 > 60)), \
                             ls[0].Select(lambda jet: jet.px()), \
                             ls[0].Select(lambda jet: jet.py()), \
                             ls[0].Select(lambda jet: jet.pz()), \
                             ls[0].Select(lambda jet: jet.rapidity()), \
                             ls[1].Select(lambda trj: trj.e()), \
                             ls[1].Select(lambda trj: trj.eta()), \
                             ls[1].Select(lambda trj: trj.index()), \
                             ls[1].Select(lambda trj: trj.m()), \
                             ls[1].Select(lambda trj: trj.phi()), \
                             ls[1].Select(lambda trj: trj.pt()), \
                             ls[1].Select(lambda trj: trj.px()), \
                             ls[1].Select(lambda trj: trj.py()), \
                             ls[1].Select(lambda trj: trj.pz()), \
                             ls[1].Select(lambda trj: trj.rapidity()), \
                             ls[2].Select(lambda bto: bto.index()), \
                             ls[2].Select(lambda bto: bto.nIP2D_TrackParticles()), \
                             ls[2].Select(lambda bto: bto.nIP3D_TrackParticles()), \
                             ls[2].Select(lambda bto: bto.nSV0_TrackParticles()), \
                             ls[2].Select(lambda bto: bto.nSV1_TrackParticles()), \
                             ls[3].Select(lambda bto: btr.index()), \
                             ls[3].Select(lambda btr: btr.nIP2D_TrackParticles()), \
                             ls[3].Select(lambda btr: btr.nIP3D_TrackParticles()), \
                             ls[3].Select(lambda btr: btr.nSV0_TrackParticles()), \
                             ls[3].Select(lambda btr: btr.nSV1_TrackParticles()), \
                             ls[4].Select(lambda clu: clu.calE()), \
                             ls[4].Select(lambda clu: clu.calEta()), \
                             ls[4].Select(lambda clu: clu.calM()), \
                             ls[4].Select(lambda clu: clu.calPhi()), \
                             ls[4].Select(lambda clu: clu.e()), \
                             ls[4].Select(lambda clu: clu.eta()), \
                             ls[4].Select(lambda clu: clu.index()), \
                             ls[4].Select(lambda clu: clu.m()), \
                             ls[4].Select(lambda clu: clu.nSamples()), \
                             ls[4].Select(lambda clu: clu.phi()), \
                             ls[4].Select(lambda clu: clu.rapidity()), \
                             ls[4].Select(lambda clu: clu.rawE()), \
                             ls[4].Select(lambda clu: clu.rawEta()), \
                             ls[4].Select(lambda clu: clu.rawM()), \
                             ls[4].Select(lambda clu: clu.rawPhi()), \
                             ls[5].Select(lambda mtp: mtp.charge()), \
                             ls[5].Select(lambda mtp: mtp.d0()), \
                             ls[5].Select(lambda mtp: mtp.e()), \
                             ls[5].Select(lambda mtp: mtp.eta()), \
                             ls[5].Select(lambda mtp: mtp.index()), \
                             ls[5].Select(lambda mtp: mtp.m()), \
                             ls[5].Select(lambda mtp: mtp.phi()), \
                             ls[5].Select(lambda mtp: mtp.pt()), \
                             ls[5].Select(lambda mtp: mtp.qOverP()), \
                             ls[5].Select(lambda mtp: mtp.rapidity()), \
                             ls[5].Select(lambda mtp: mtp.theta()), \
                             ls[5].Select(lambda mtp: mtp.vz()), \
                             ls[5].Select(lambda mtp: mtp.z0()), \
                             ls[6].Select(lambda ele: ele.charge()), \
                             ls[6].Select(lambda ele: ele.e()), \
                             ls[6].Select(lambda ele: ele.eta()), \
                             ls[6].Select(lambda ele: ele.index()), \
                             ls[6].Select(lambda ele: ele.m()), \
                             ls[6].Select(lambda ele: ele.nCaloClusters()), \
                             ls[6].Select(lambda ele: ele.nTrackParticles()), \
                             ls[6].Select(lambda ele: ele.phi()), \
                             ls[6].Select(lambda ele: ele.pt()), \
                             ls[6].Select(lambda ele: ele.rapidity()), \
                             ls[7].Select(lambda gsf: gsf.charge()), \
                             ls[7].Select(lambda gsf: gsf.d0()), \
                             ls[7].Select(lambda gsf: gsf.e()), \
                             ls[7].Select(lambda gsf: gsf.eta()), \
                             ls[7].Select(lambda gsf: gsf.index()), \
                             ls[7].Select(lambda gsf: gsf.m()), \
                             ls[7].Select(lambda gsf: gsf.phi()), \
                             ls[7].Select(lambda gsf: gsf.pt()), \
                             ls[7].Select(lambda gsf: gsf.qOverP()), \
                             ls[7].Select(lambda gsf: gsf.rapidity()), \
                             ls[7].Select(lambda gsf: gsf.theta()), \
                             ls[7].Select(lambda gsf: gsf.vz()), \
                             ls[7].Select(lambda gsf: gsf.z0()), \
                             ls[8].Select(lambda muo: muo.charge()), \
                             ls[8].Select(lambda muo: muo.e()), \
                             ls[8].Select(lambda muo: muo.eta()), \
                             ls[8].Select(lambda muo: muo.index()), \
                             ls[8].Select(lambda muo: muo.m()), \
                             ls[8].Select(lambda muo: muo.nMuonSegments()), \
                             ls[8].Select(lambda muo: muo.phi()), \
                             ls[8].Select(lambda muo: muo.pt()), \
                             ls[8].Select(lambda muo: muo.quality()), \
                             ls[8].Select(lambda muo: muo.rapidity()), \
                             ls[9].Select(lambda ptn: ptn.conversionRadius()), \
                             ls[9].Select(lambda ptn: ptn.e()), \
                             ls[9].Select(lambda ptn: ptn.eta()), \
                             ls[9].Select(lambda ptn: ptn.index()), \
                             ls[9].Select(lambda ptn: ptn.m()), \
                             ls[9].Select(lambda ptn: ptn.nCaloClusters()), \
                             ls[9].Select(lambda ptn: ptn.nVertices()), \
                             ls[9].Select(lambda ptn: ptn.phi()), \
                             ls[9].Select(lambda ptn: ptn.pt()), \
                             ls[9].Select(lambda ptn: ptn.rapidity()))') \
        .AsAwkwardArray(("JetE", \
                         "JetEta", \
                         "JetIndex", \
                         "JetM", \
                         "JetPhi", \
                         "JetPt", \
                         "JetPx", \
                         "JetPy", \
                         "JetPz", \
                         "JetRapidity", \
                         "TrackE", \
                         "TrackEta", \
                         "TrackIndex", \
                         "TrackM", \
                         "TrackPhi", \
                         "TrackPt", \
                         "TrackPx", \
                         "TrackPy", \
                         "TrackPz", \
                         "TrackRapidity", \
                         "BTopoIndex", \
                         "BTopo2DTrack", \
                         "BTopo3DTrack", \
                         "BTopoSV0", \
                         "BTopoSV1", \
                         "BTrackIndex", \
                         "BTrack2DTrack", \
                         "BTrack3DTrack", \
                         "BTrackSV0", \
                         "BTrackSV1", \
                         "ClustersCalE", \
                         "ClustersCalEta", \
                         "ClustersCalM", \
                         "ClustersCalPhi", \
                         "ClustersE", \
                         "ClustersEta", \
                         "ClustersIndex", \
                         "ClustersM", \
                         "ClustersNSamples", \
                         "ClustersPhi", \
                         "ClustersRapidity", \
                         "ClustersRawE", \
                         "ClustersRawEta", \
                         "ClustersRawM", \
                         "ClustersRawPhi", \
                         "MuonTrackCharge", \
                         "MuonTrackD0", \
                         "MuonTrackE", \
                         "MuonTrackEta", \
                         "MuonTrackIndex", \
                         "MuonTrackM", \
                         "MuonTrackPhi", \
                         "MuonTrackPt", \
                         "MuonTrackQOverP", \
                         "MuonTrackRapidity", \
                         "MuonTrackTheta", \
                         "MuonTrackVZ", \
                         "MuonTrackZ0", \
                         "EleCharge", \
                         "EleE", \
                         "EleEta", \
                         "EleIndex", \
                         "EleM", \
                         "EleNClusters", \
                         "EleNTrack", \
                         "ElePhi", \
                         "ElePt", \
                         "EleRapidity", \
                         "GSFCharge", \
                         "GSFD0", \
                         "GSFE", \
                         "GSFEta", \
                         "GSFIndex", \
                         "GSFM", \
                         "GSFPhi", \
                         "GSFPt", \
                         "GSFQOverP", \
                         "GSFRapidity", \
                         "GSFTheta", \
                         "GSFVZ", \
                         "GSFZ0", \
                         "MuonCharge", \
                         "MuonE", \
                         "MuonEta", \
                         "MuonIndex", \
                         "MuonM", \
                         "MuonN", \
                         "MuonPhi", \
                         "MuonPt", \
                         "MuonQuality", \
                         "MuonRapidity", \
                         "PhotonRadius", \
                         "PhotonE", \
                         "PhotonEta", \
                         "PhotonIndex", \
                         "PhotonM", \
                         "PhotonNClusters", \
                         "PhotonNVertices", \
                         "PhotonPhi", \
                         "PhotonPt", \
                         "PhotonRapidity")) \
        .value()

    retrieved_data = query[b'JetPt'] # store the JetPts as a list

    assert len(retrieved_data) == 5000 # did we retrieve the correct number of JetPts?
	
# This test retrieves 20 columns from 10 datasets at the same time. This takes a lot of memory!
@pytest.mark.asyncio
@pytest.mark.stress
async def test_multiple_requests():
    dataset = [ServiceXDataset('mc15_13TeV:mc15_13TeV.361106.PowhegPythia8EvtGen_AZNLOCTEQ6L1_Zee.merge.DAOD_STDM3.e3601_s2576_s2132_r6630_r6264_p2363_tid05630052_00'), \
               ServiceXDataset('mc16_13TeV:mc16_13TeV.301000.PowhegPythia8EvtGen_AZNLOCTEQ6L1_DYee_120M180.deriv.DAOD_SUSY18.e3649_e5984_s3126_r10201_r10210_p3840_tid18281770_00'), \
               ServiceXDataset('mc16_13TeV:mc16_13TeV.301000.PowhegPythia8EvtGen_AZNLOCTEQ6L1_DYee_120M180.deriv.DAOD_EXOT0.e3649_e5984_s3126_r10724_r10726_p4180_tid21859882_00'), \
               ServiceXDataset('mc16_13TeV:mc16_13TeV.301000.PowhegPythia8EvtGen_AZNLOCTEQ6L1_DYee_120M180.deriv.DAOD_EXOT0.e3649_e5984_s3126_r10724_r10726_p4180_tid21859885_00'), \
               ServiceXDataset('mc16_13TeV:mc16_13TeV.301000.PowhegPythia8EvtGen_AZNLOCTEQ6L1_DYee_120M180.deriv.DAOD_EXOT0.e3649_e5984_s3126_r9364_r9315_p4180_tid21859934_00'), \
               ServiceXDataset('mc16_13TeV:mc16_13TeV.301000.PowhegPythia8EvtGen_AZNLOCTEQ6L1_DYee_120M180.deriv.DAOD_EXOT0.e3649_s3126_r10201_r10210_p4180_tid21860366_00'), \
               ServiceXDataset('mc16_13TeV:mc16_13TeV.301000.PowhegPythia8EvtGen_AZNLOCTEQ6L1_DYee_120M180.deriv.DAOD_EXOT12.e3649_e5984_s3126_r10724_r10726_p3978_tid19368338_00'), \
               ServiceXDataset('mc16_13TeV:mc16_13TeV.301001.PowhegPythia8EvtGen_AZNLOCTEQ6L1_DYee_180M250.deriv.DAOD_EXOT19.e3649_e5984_s3126_r10724_r10726_p3978_tid19509568_00'), \
               ServiceXDataset('mc16_13TeV:mc16_13TeV.301000.PowhegPythia8EvtGen_AZNLOCTEQ6L1_DYee_120M180.deriv.DAOD_EXOT12.e3649_e5984_s3126_r9364_r9315_p3978_tid19370588_00'), \
               ServiceXDataset('mc16_13TeV:mc16_13TeV.301000.PowhegPythia8EvtGen_AZNLOCTEQ6L1_DYee_120M180.deriv.DAOD_EXOT19.e3649_e5984_s3126_r10724_r10726_p3978_tid19511379_00') \
               ]

    async def fetch_data(dataset):
        query = ServiceXDatasetSource(dataset) \
            .Select('lambda e: (e.Jets("AntiKt4EMTopoJets"), e.Electrons("Electrons"))') \
            .Select('lambda ls: (ls[0].Select(lambda jet: jet.e()), \
                                 ls[0].Select(lambda jet: jet.eta()), \
                                 ls[0].Select(lambda jet: jet.index()), \
                                 ls[0].Select(lambda jet: jet.m()), \
                                 ls[0].Select(lambda jet: jet.phi()), \
                                 ls[0].Select(lambda jet: jet.pt()), \
                                 ls[0].Select(lambda jet: jet.px()), \
                                 ls[0].Select(lambda jet: jet.py()), \
                                 ls[0].Select(lambda jet: jet.pz()), \
                                 ls[0].Select(lambda jet: jet.rapidity()), \
                                 ls[1].Select(lambda ele: ele.charge()), \
                                 ls[1].Select(lambda ele: ele.e()), \
                                 ls[1].Select(lambda ele: ele.eta()), \
                                 ls[1].Select(lambda ele: ele.index()), \
                                 ls[1].Select(lambda ele: ele.m()), \
                                 ls[1].Select(lambda ele: ele.nCaloClusters()), \
                                 ls[1].Select(lambda ele: ele.nTrackParticles()), \
                                 ls[1].Select(lambda ele: ele.phi()), \
                                 ls[1].Select(lambda ele: ele.pt()), \
                                 ls[1].Select(lambda ele: ele.rapidity()))') \
            .AsAwkwardArray(("JetE", \
                             "JetEta", \
                             "JetIndex", \
                             "JetM", \
                             "JetPhi", \
                             "JetPt", \
                             "JetPx", \
                             "JetPy", \
                             "JetPz", \
                             "JetRapidity", \
                             "EleCharge", \
                             "EleE", \
                             "EleEta", \
                             "EleIndex", \
                             "EleM", \
                             "EleNClusters", \
                             "EleNTrack", \
                             "ElePhi", \
                             "ElePt", \
                             "EleRapidity")) \
            .value_async()

        return await query

    data_list = []
    for i in range(10):
        data_slot = fetch_data(dataset[i])
        data_list.append(data_slot)

    ds0, ds1, ds2, ds3, ds4, ds5, ds6, ds7, ds8, ds9 = await asyncio.gather(data_list[0], \
                                                            data_list[1], \
                                                            data_list[2], \
                                                            data_list[3], \
                                                            data_list[4], \
                                                            data_list[5], \
                                                            data_list[6], \
                                                            data_list[7], \
                                                            data_list[8], \
                                                            data_list[9])

    total_length = len(ds0[b'JetPt']) + len(ds1[b'JetPt']) + \
                   len(ds2[b'JetPt']) + len(ds3[b'JetPt']) + \
                   len(ds4[b'JetPt']) + len(ds5[b'JetPt']) + \
                   len(ds6[b'JetPt']) + len(ds7[b'JetPt']) + \
                   len(ds8[b'JetPt']) + len(ds9[b'JetPt'])

# did we pull the correct amount of data from all 10 datasets?
    assert total_length == 10264173