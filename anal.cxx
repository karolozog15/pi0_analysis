// =============================================================================
// anal_pi0.C - rekonstrukcja pi0 -> 2 gamma (bez rho i bez pionu naladowanego)
//
//  Grupy kombinacji; KAZDA NASTEPNA jest odpalana tylko gdy poprzednia zawiodla
//  (brak pi0 z masa w oknie winLo*m(pi0) < m < winHi*m(pi0)):
//    1. P+P                 - pary potential gamma (charge==0)                       [0: PP]
//    2. split P             - split klastra potential gamma, TYLKO gdy jest dokladnie
//                             1 potential gamma                                      [1: SplitP]
//    3. P+I, I+I            - potential x improved, pary improved                    [2: PI, 3: II]
//                             (improved = klaster nieuzyty przez czastke reko i BEZ
//                              dopasowanego sladu - TrackClusterMatch)
//    4. split I             - split kazdego improved klastra na 2                    [4: SplitI]
//    5. P+H, I+H, H+H       - potential / improved / nowe klastry z wolnych hitow    [5: PH, 6: IH, 7: HH]
//                             (nowy klaster: > nHitsMin hitow w dR<hitDR)
//  W obrebie grupy wybierana jest kombinacja o masie NAJBLIZSZEJ m(pi0).
//
//  Wierzcholek: CentralTrackVertices (najwiecej powiazanych czastek), brak -> (0,0,0).
//  Truth klastrow: *ClusterAssociations (getRec = klaster, getSim = czastka MC).
//
//  Histogramy 2D MC (x) vs reko (y) tylko dla final pi0 w oknie masy:
//   - fotony: E, eta, phi  x  klasa (P, I, S, H)  x  technologia (barrel, imaging, scfi, endcapP, endcapN)
//   - pi0:    E, eta, phi  x  grupa (PP, SplitP, PI_II, SplitI, PH_IH_HH)
// =============================================================================
#include <edm4hep/MCParticleCollection.h>
#include <edm4eic/ClusterCollection.h>
#include <edm4eic/CalorimeterHitCollection.h>
#include <edm4eic/MCRecoClusterParticleAssociationCollection.h>
#include <edm4eic/TrackClusterMatchCollection.h>
#include <edm4eic/ReconstructedParticleCollection.h>
#include <edm4eic/VertexCollection.h>

#include <podio/Frame.h>
#include <podio/ROOTReader.h>
#include <podio/ObjectID.h>

#include <TFile.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TVector3.h>
#include <TLorentzVector.h>

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <cmath>
#include <filesystem>

typedef std::pair<uint32_t, uint32_t> IDKey;
typedef std::set<IDKey> IDSet;
static IDKey keyOf(const podio::ObjectID& id) { return {id.collectionID, id.index}; }

static const double kMpi0 = 0.13498;
static const int    kNM   = 8;   // metody: PP, SplitP, PI, II, SplitI, PH, IH, HH
static const int    kNS   = 5;   // grupy / etapy (takze grupy pi0 w histogramach 2D)
static const int    kNC   = 8;   // kategorie pochodzenia klastra
static const int    kNCls = 4;   // klasy fotonow: 0 P, 1 I, 2 S, 3 H
static const int    kNTech = 5;  // technologie: 0 barrel, 1 imaging, 2 scfi, 3 endcapP, 4 endcapN

struct PhI  { TLorentzVector p4; int cls = -1; int tech = -1; };   // foton kandydat: 4-pedy, klasa (algorytm), technologia (-1 = nieznana)
struct Gam  { PhI ph; edm4eic::ReconstructedParticle rec; };       // potential gamma: foton (klasa P) + czastka reko
struct Hit  { TVector3 pos; double e; };
struct ClInfo { double w; int cat; int pdg; };                     // najlepsza asocjacja klastra: waga, kategoria, PDG czastki MC

// srodek ciezkosci hitow wazony energia; e = suma energii
static Hit centroid(const std::vector<Hit>& g) {
  TVector3 c; double s = 0;
  for (const auto& h : g) { c += h.e * h.pos; s += h.e; }
  if (s > 0) c *= 1.0 / s;
  return {c, s};
}
// klaster + podklastry (2 poziomy w glab) -> zbior ID
static void markCluster(const edm4eic::Cluster& c, IDSet& used, int depth = 0) {
  if (!c.isAvailable()) return;
  used.insert(keyOf(c.getObjectID()));
  if (depth < 2) for (const auto& sub : c.getClusters()) markCluster(sub, used, depth + 1);
}
// wszystkie hity klastra i podklastrow (bez duplikatow)
static void gatherHits(const edm4eic::Cluster& c, std::vector<Hit>& out, IDSet& seen, int depth = 0) {
  if (!c.isAvailable()) return;
  for (const auto& h : c.getHits()) {
    if (!h.isAvailable() || !seen.insert(keyOf(h.getObjectID())).second) continue;
    out.push_back({TVector3(h.getPosition().x, h.getPosition().y, h.getPosition().z), h.getEnergy()});
  }
  if (depth < 2) for (const auto& sub : c.getClusters()) gatherHits(sub, out, seen, depth + 1);
}
// split hitow klastra na 2 grupy (wazony k-means): seed A = najsilniejszy hit, seed B = najsilniejszy hit
// oddalony o > minSep i niosacy >= minFrac energii
static bool split2(const std::vector<Hit>& hs, std::vector<Hit>& g1, std::vector<Hit>& g2, double minSep, double minFrac) {
  if (hs.size() < 4) return false;
  const double tot = centroid(hs).e;
  size_t iA = 0;
  for (size_t i = 1; i < hs.size(); ++i) if (hs[i].e > hs[iA].e) iA = i;
  int iB = -1;
  for (size_t k = 0; k < hs.size(); ++k)
    if (hs[k].e >= minFrac * tot && hs[k].pos.DeltaR(hs[iA].pos) > minSep && (iB < 0 || hs[k].e > hs[iB].e)) iB = (int)k;
  if (iB < 0) return false;
  Hit c1 = hs[iA], c2 = hs[iB];
  for (int it = 0; it < 10; ++it) {
    g1.clear(); g2.clear();
    for (const auto& h : hs) (h.pos.DeltaR(c1.pos) <= h.pos.DeltaR(c2.pos) ? g1 : g2).push_back(h);
    if (g1.empty() || g2.empty()) return false;
    c1 = centroid(g1); c2 = centroid(g2);
  }
  return c1.e >= minFrac * tot && c2.e >= minFrac * tot;
}

// =============================================================================
void anal(const std::string& out_file = "pi0_improved.root",
          double E_CUT = 0.1,                       // min. energia fotonu MC / klastra
          long   nMax = 0,                          // 0 = wszystkie eventy
          double winLo = 0.75, double winHi = 1.25, // okno masy: winLo*m(pi0) < m < winHi*m(pi0)
          double splitSep = 0.16, double splitFrac = 0.01,
          double hitDR = 0.2, int nHitsMin = 4) {

  // ---------------- pliki wejsciowe ----------------
  const std::string in_dir = "/run/media/epic/Data/PI0/DVMP_pi0_9x130_26.07";
  const std::string prefix = "DVMP_pi0_9x130";
  std::vector<std::string> in_file;
  if (std::filesystem::exists(in_dir))
    for (const auto& entry : std::filesystem::directory_iterator(in_dir)) {
      const std::string fn = entry.path().filename().string();
      if (entry.is_regular_file() && entry.path().extension() == ".root" && fn.rfind(prefix, 0) == 0)
        in_file.push_back(entry.path().string());
    }
  std::sort(in_file.begin(), in_file.end());
  if (in_file.empty()) { std::cerr << "Brak plikow: " << prefix << "*.root w " << in_dir << "\n"; return; }

  podio::ROOTReader reader;
  reader.openFiles(in_file);
  TFile* fout = new TFile(out_file.c_str(), "RECREATE");

  const std::vector<std::string> clNames  = {"EcalBarrelClusters", "EcalBarrelScFiClusters", "EcalBarrelImagingClusters",
                                             "EcalEndcapPClusters", "EcalEndcapNClusters"};
  const std::vector<std::string> hitNames = {"EcalBarrelScFiRecHits", "EcalBarrelImagingRecHits",
                                             "EcalEndcapPRecHits", "EcalEndcapNRecHits"};
  const std::vector<std::string> clAssocNames = {"EcalBarrelClusterAssociations", "EcalBarrelScFiClusterAssociations",
                                                 "EcalBarrelImagingClusterAssociations",
                                                 "EcalEndcapPClusterAssociations", "EcalEndcapNClusterAssociations"};
  const std::vector<std::string> tcmNames = {"EcalBarrelTrackClusterMatches", "EcalEndcapPTrackClusterMatches",
                                             "EcalEndcapNTrackClusterMatches"};
  // technologia: 0 barrel, 1 imaging, 2 scfi, 3 endcapP, 4 endcapN
  const int techOfCl[5]  = {0, 2, 1, 3, 4};   // wg clNames:  barrel, scfi, imaging, endcapP, endcapN
  const int techOfHit[4] = {2, 1, 3, 4};      // wg hitNames: scfi, imaging, endcapP, endcapN
  const int grpOfMethod[kNM] = {0, 1, 2, 2, 3, 4, 4, 4};   // metoda -> grupa pi0: PP | SplitP | PI+II | SplitI | PH+IH+HH

  // ---------------- histogramy ----------------
  TH1D* mcPi0_pt    = new TH1D("mcPi0_pt",    "MC #pi^{0};p_{T} [GeV];N", 30, 0, 6);
  TH1D* mcPi0_eta   = new TH1D("mcPi0_eta",   "MC #pi^{0};#eta;N",        100, -5, 5);
  TH1D* mcPi0_E     = new TH1D("mcPi0_E",     "MC #pi^{0};E [GeV];N",     40, 0, 10);
  TH1D* mcPi0_m     = new TH1D("mcPi0_m",     "MC #pi^{0};m [GeV];N",     100, 0, 0.5);
  TH1D* mcGamma_pt  = new TH1D("mcGamma_pt",  "MC #gamma;p_{T} [GeV];N",  30, 0, 6);
  TH1D* mcGamma_eta = new TH1D("mcGamma_eta", "MC #gamma;#eta;N",         100, -5, 5);
  TH1D* mcGamma_E   = new TH1D("mcGamma_E",   "MC #gamma;E [GeV];N",      50, 0, 10);
  TH1D* finalPi0_pt    = new TH1D("finalPi0_pt",    "final #pi^{0};p_{T} [GeV];N", 30, 0, 6);
  TH1D* finalPi0_eta   = new TH1D("finalPi0_eta",   "final #pi^{0};#eta;N",        100, -5, 5);
  TH1D* finalPi0_E     = new TH1D("finalPi0_E",     "final #pi^{0};E [GeV];N",     40, 0, 10);
  TH1D* finalPi0_m     = new TH1D("finalPi0_m",     "final #pi^{0};m [GeV];N",     100, 0, 0.5);
  TH1D* finalPi0_dR    = new TH1D("finalPi0_dR",    "final #pi^{0};dR(final, MC #pi^{0});N", 100, 0, 1.5);
  TH1D* finalGamma_pt  = new TH1D("finalGamma_pt",  "final #gamma;p_{T} [GeV];N",  30, 0, 6);
  TH1D* finalGamma_eta = new TH1D("finalGamma_eta", "final #gamma;#eta;N",         100, -5, 5);
  TH1D* finalGamma_E   = new TH1D("finalGamma_E",   "final #gamma;E [GeV];N",      50, 0, 10);
  TH1D* finalGamma_dR  = new TH1D("finalGamma_dR",  "final #gamma;dR(final, nearest MC #gamma);N", 100, 0, 1.5);
  TH1D* potPi0_m  = new TH1D("potPi0_m",  "wszystkie pary potential #gamma;m [GeV];N", 100, 0, 0.5);
  TH1D* h_nPot    = new TH1D("h_nPot",    "Liczba potential gamma w evencie;N;events", 11, -0.5, 10.5);

  const char* methodLabel[kNM] = {"P+P", "Split P (1 gamma)", "P+I", "I+I", "Split I", "P+H", "I+H", "H+H"};
  const char* methodTag[kNM]   = {"PP", "SplitP", "PI", "II", "SplitI", "PH", "IH", "HH"};
  TH1D* h_method     = new TH1D("h_method",     "Metoda finalnego pi0 (w oknie);;pi0", kNM, 0, kNM);
  TH1D* h_methodGood = new TH1D("h_methodGood", "Metoda finalnego pi0 (w oknie, zgodne z MC);;pi0", kNM, 0, kNM);

  // rozdzielczosci (MC - reko)
  TH1D* h_pi_E_res    = new TH1D("h_pi_E_res",    ";(E_{MC}-E_{rec})/E_{MC};N",  100, -1, 1);
  TH1D* h_pi_eta_res  = new TH1D("h_pi_eta_res",  ";#eta_{MC}-#eta_{rec};N",     100, -0.5, 0.5);
  TH1D* h_gam_E_res   = new TH1D("h_gam_E_res",   ";(E_{MC}-E_{rec})/E_{MC};N",  100, -1, 1);
  TH1D* h_gam_eta_res = new TH1D("h_gam_eta_res", ";#eta_{MC}-#eta_{rec};N",     100, -0.5, 0.5);

  // wierzcholek i liczba dodatkowych klastrow
  TH1D* h_vtx_z = new TH1D("h_vtx_z", ";vertex z [mm];N", 100, -200, 200);
  TH1D* h_vtx_x = new TH1D("h_vtx_x", ";vertex x [mm];N", 100, -50, 50);
  TH1D* h_vtx_y = new TH1D("h_vtx_y", ";vertex y [mm];N", 100, -50, 50);
  TH1D* h_nVtx  = new TH1D("h_nVtx",  "Liczba wierzcholkow (CentralTrackVertices);N;events", 11, -0.5, 10.5);
  TH1D* h_nImproved    = new TH1D("h_nImproved",    "improved klastry (bez sladu) w evencie;N;events", 21, -0.5, 20.5);
  TH1D* h_nImprovedTrk = new TH1D("h_nImprovedTrk", "improved klastry odrzucone (maja slad) w evencie;N;events", 21, -0.5, 20.5);
  TH1D* h_nImpTrue     = new TH1D("h_nImpTrue",     "improved klastry od #gamma z #pi^{0} (truth) w evencie;N;events", 5, -0.5, 4.5);
  TH1D* h_nNewCl       = new TH1D("h_nNewCl",       "nowe klastry z hitow w evencie (zdarzenia z grupy 5);N;events", 51, -0.5, 50.5);
  TH1D* h_nCand[kNS];
  for (int s = 0; s < kNS; ++s)
    h_nCand[s] = new TH1D(Form("h_nCand_g%d", s + 1), Form("grupa %d;liczba kombinacji w evencie;events", s + 1), 100, 0, 1000);

  // pochodzenie klastrow (truth)
  const char* catLabel[kNC] = {"#gamma from #pi^{0}", "other #gamma", "e^{#pm}", "#mu^{#pm}", "#pi^{#pm}", "n/#bar{n}", "other", "no assoc."};
  TH1D* h_impCat    = new TH1D("h_impCat",    "improved klastry (zostawione): pochodzenie MC;;klastry", kNC, 0, kNC);
  TH1D* h_impCatTrk = new TH1D("h_impCatTrk", "improved klastry odrzucone (slad): pochodzenie MC;;klastry", kNC, 0, kNC);
  TH1D* h_potCat    = new TH1D("h_potCat",    "klaster potential gamma: pochodzenie MC;;potential gamma", kNC, 0, kNC);
  for (int i = 0; i < kNC; ++i) {
    h_impCat->GetXaxis()->SetBinLabel(i + 1, catLabel[i]);
    h_impCatTrk->GetXaxis()->SetBinLabel(i + 1, catLabel[i]);
    h_potCat->GetXaxis()->SetBinLabel(i + 1, catLabel[i]);
  }

  // final pi0 / final fotony osobno dla metody k (tylko dla wybranego final pi0)
  TH1D *fPi0_E[kNM], *fPi0_eta[kNM], *fPi0_m[kNM], *fPi0_dR[kNM];
  TH1D *fGam_E[kNM], *fGam_eta[kNM], *fGam_dR[kNM];
  for (int k = 0; k < kNM; ++k) {
    fPi0_E[k]   = new TH1D(Form("final_%s_pi0_E", methodTag[k]),   Form("%s: final #pi^{0};E [GeV];N", methodLabel[k]),   40, 0, 10);
    fPi0_eta[k] = new TH1D(Form("final_%s_pi0_eta", methodTag[k]), Form("%s: final #pi^{0};#eta;N", methodLabel[k]),       100, -5, 5);
    fPi0_m[k]   = new TH1D(Form("final_%s_pi0_m", methodTag[k]),   Form("%s: final #pi^{0};m [GeV];N", methodLabel[k]),   100, 0, 0.5);
    fPi0_dR[k]  = new TH1D(Form("final_%s_pi0_dR", methodTag[k]),  Form("%s: final #pi^{0};dR(final, MC #pi^{0});N", methodLabel[k]), 100, 0, 1.5);
    fGam_E[k]   = new TH1D(Form("final_%s_gam_E", methodTag[k]),   Form("%s: final #gamma;E [GeV];N", methodLabel[k]),    50, 0, 10);
    fGam_eta[k] = new TH1D(Form("final_%s_gam_eta", methodTag[k]), Form("%s: final #gamma;#eta;N", methodLabel[k]),        100, -5, 5);
    fGam_dR[k]  = new TH1D(Form("final_%s_gam_dR", methodTag[k]),  Form("%s: final #gamma;dR(final, nearest MC #gamma);N", methodLabel[k]), 100, 0, 1.5);
    h_method->GetXaxis()->SetBinLabel(k + 1, methodLabel[k]);
    h_methodGood->GetXaxis()->SetBinLabel(k + 1, methodLabel[k]);
  }

  // histogramy 2D MC (x) vs reko (y): fotony klasa x technologia, pi0 grupy
  const char* clsTag[kNCls]   = {"P", "I", "S", "H"};
  const char* clsLabel[kNCls] = {"P: czastka reko", "I: improved klaster", "S: split", "H: hity"};
  const char* techTag[kNTech] = {"barrel", "imaging", "scfi", "endcapP", "endcapN"};
  TH2D *h2_gam_E[kNCls][kNTech], *h2_gam_eta[kNCls][kNTech], *h2_gam_phi[kNCls][kNTech];
  for (int c = 0; c < kNCls; ++c)
    for (int t = 0; t < kNTech; ++t) {
      const char* ttl = Form("%s, %s", clsLabel[c], techTag[t]);
      h2_gam_E[c][t]   = new TH2D(Form("h2_gam_E_%s_%s", clsTag[c], techTag[t]),
                                  Form("%s;E_{MC} [GeV];E_{reco} [GeV]", ttl), 50, 0, 20, 50, 0, 20);
      h2_gam_eta[c][t] = new TH2D(Form("h2_gam_eta_%s_%s", clsTag[c], techTag[t]),
                                  Form("%s;#eta_{MC};#eta_{reco}", ttl), 100, -5, 5, 100, -5, 5);
      h2_gam_phi[c][t] = new TH2D(Form("h2_gam_phi_%s_%s", clsTag[c], techTag[t]),
                                  Form("%s;#phi_{MC} [rad];#phi_{reco} [rad]", ttl), 60, -M_PI, M_PI, 60, -M_PI, M_PI);
    }
  const char* grpTag[kNS]   = {"PP", "SplitP", "PI_II", "SplitI", "PH_IH_HH"};
  const char* grpLabel[kNS] = {"1: P+P", "2: split P (1 gamma)", "3: P+I, I+I", "4: split I", "5: P+H, I+H, H+H"};
  TH2D *h2_pi0_E[kNS], *h2_pi0_eta[kNS], *h2_pi0_phi[kNS];
  for (int g = 0; g < kNS; ++g) {
    h2_pi0_E[g]   = new TH2D(Form("h2_pi0_E_%s", grpTag[g]),   Form("#pi^{0}, %s;E_{MC} [GeV];E_{reco} [GeV]", grpLabel[g]),
                             50, 0, 40, 50, 0, 40);
    h2_pi0_eta[g] = new TH2D(Form("h2_pi0_eta_%s", grpTag[g]), Form("#pi^{0}, %s;#eta_{MC};#eta_{reco}", grpLabel[g]),
                             100, -5, 5, 100, -5, 5);
    h2_pi0_phi[g] = new TH2D(Form("h2_pi0_phi_%s", grpTag[g]), Form("#pi^{0}, %s;#phi_{MC} [rad];#phi_{reco} [rad]", grpLabel[g]),
                             60, -M_PI, M_PI, 60, -M_PI, M_PI);
  }

  long nEv = 0, nPi0 = 0, nVtxEv = 0;
  long nWin[kNS] = {}, nGood[kNS] = {};
  long nGamFill[kNCls][kNTech] = {};
  long nGamNoTech[kNCls] = {};
  long nPi0Grp[kNS] = {};
  std::map<int, long> pdgImp, pdgImpTrk;   // PDG czastek MC dla improved klastrow nie-gamma-z-pi0 (zostawione / odrzucone)

  while (auto entry = reader.readNextEntry(podio::Category::Event)) {
    ++nEv;
    if (nMax > 0 && nEv > nMax) break;
    if (nEv % 10000 == 0) std::cout << "Przetworzono " << nEv << " zdarzen.\n";
    auto frame = podio::Frame(std::move(entry));

    const auto& mc    = frame.get<edm4hep::MCParticleCollection>("MCParticles");
    const auto& recos = frame.get<edm4eic::ReconstructedParticleCollection>("ReconstructedParticles");

    // ================= MC: pi0 i jego 2 fotony =================
    TLorentzVector pi0MC;
    bool havePi0 = false;
    std::vector<TLorentzVector> photons;
    for (const auto& p : mc) {
      if (!havePi0 && p.getPDG() == 111 && p.getGeneratorStatus() == 2) {
        pi0MC.SetXYZM(p.getMomentum().x, p.getMomentum().y, p.getMomentum().z, p.getMass());
        havePi0 = true;
      }
      if (p.getPDG() != 22 || p.getGeneratorStatus() != 1) continue;
      bool fromPi0 = false;
      for (const auto& par : p.getParents()) if (par.getPDG() == 111) fromPi0 = true;
      if (!fromPi0) continue;
      TLorentzVector ph;
      ph.SetXYZM(p.getMomentum().x, p.getMomentum().y, p.getMomentum().z, 0.0);
      if (ph.E() < E_CUT) continue;
      if (1.3 < std::fabs(ph.Eta()) && std::fabs(ph.Eta()) < 1.6) continue;
      photons.push_back(ph);
    }
    if (!havePi0 || photons.size() != 2) continue;

    ++nPi0;
    mcPi0_pt->Fill(pi0MC.Pt()); mcPi0_eta->Fill(pi0MC.Eta()); mcPi0_E->Fill(pi0MC.E()); mcPi0_m->Fill(pi0MC.M());
    for (const auto& g : photons) { mcGamma_pt->Fill(g.Pt()); mcGamma_eta->Fill(g.Eta()); mcGamma_E->Fill(g.E()); }

    // ================= technologia kazdego klastra (wg kolekcji) =================
    std::map<IDKey, int> clTech;
    for (size_t i = 0; i < clNames.size(); ++i)
      for (const auto& c : frame.get<edm4eic::ClusterCollection>(clNames[i])) clTech[keyOf(c.getObjectID())] = techOfCl[i];

    // ================= potential gamma (charge==0): pedy czastki reko, technologia z jej klastrow =================
    std::vector<Gam> gams;
    for (const auto& rp : recos) {
      if (std::fabs(rp.getCharge()) > 0.5) continue;
      TLorentzVector g;
      g.SetXYZM(rp.getMomentum().x, rp.getMomentum().y, rp.getMomentum().z, 0.0);
      if (g.E() <= 0) continue;
      // technologia: klaster + podklastry czastki; imaging i scfi razem (albo barrel scalony) = barrel
      IDSet ks;
      for (const auto& c : rp.getClusters()) markCluster(c, ks);
      bool bar = false, img = false, sci = false;
      int other = -1;
      for (const auto& k : ks) {
        auto it = clTech.find(k);
        if (it == clTech.end()) continue;
        if (it->second == 0) bar = true; else if (it->second == 1) img = true; else if (it->second == 2) sci = true; else other = it->second;
      }
      const int tech = (bar || (img && sci)) ? 0 : img ? 1 : sci ? 2 : other;   // other: endcapP / endcapN albo -1
      gams.push_back({PhI{g, 0, tech}, rp});
    }
    h_nPot->Fill(gams.size());

    // ================= wierzcholek z CentralTrackVertices (najwiecej powiazanych czastek); brak -> (0,0,0) =================
    TVector3 vtx(0, 0, 0);
    {
      const auto& verts = frame.get<edm4eic::VertexCollection>("CentralTrackVertices");
      h_nVtx->Fill(verts.size());
      long bestN = -1;
      for (const auto& v : verts) {
        const long n = v.getAssociatedParticles().size();
        if (n > bestN) { bestN = n; vtx.SetXYZ(v.getPosition().x, v.getPosition().y, v.getPosition().z); }
      }
      if (bestN >= 0) { ++nVtxEv; h_vtx_z->Fill(vtx.Z());h_vtx_x->Fill(vtx.X());h_vtx_y->Fill(vtx.Y()); }
    }
    // foton bezmasowy o energii E i kierunku (pos - wierzcholek)
    auto mkPh = [&](const TVector3& pos, double E) {
      const TVector3 u = (pos - vtx).Unit();
      return TLorentzVector(E * u.X(), E * u.Y(), E * u.Z(), E);
    };

    // ================= truth klastrow: najlepsza asocjacja klaster -> czastka MC =================
    std::map<IDKey, ClInfo> clInfo;
    for (const auto& name : clAssocNames)
      for (const auto& a : frame.get<edm4eic::MCRecoClusterParticleAssociationCollection>(name)) {
        if (!a.getRec().isAvailable() || !a.getSim().isAvailable()) continue;
        const IDKey rk = keyOf(a.getRec().getObjectID());
        auto it = clInfo.find(rk);
        if (it != clInfo.end() && a.getWeight() <= it->second.w) continue;
        const auto sim = a.getSim();
        const int apdg = std::abs(sim.getPDG());
        int cat = 6;   // 0 gamma z pi0, 1 inny gamma, 2 e, 3 mu, 4 pi, 5 n/nbar, 6 inne, 7 brak asocjacji
        if (apdg == 22) { cat = 1; for (const auto& par : sim.getParents()) if (par.getPDG() == 111) cat = 0; }
        else if (apdg == 11)   cat = 2;
        else if (apdg == 13)   cat = 3;
        else if (apdg == 211)  cat = 4;
        else if (apdg == 2112) cat = 5;
        clInfo[rk] = {a.getWeight(), cat, sim.getPDG()};
      }
    // pochodzenie klastra potential gamma (pierwszy dostepny klaster)
    for (const auto& g : gams)
      for (const auto& c : g.rec.getClusters())
        if (c.isAvailable()) {
          auto it = clInfo.find(keyOf(c.getObjectID()));
          h_potCat->Fill(it != clInfo.end() ? it->second.cat : 7);
          break;
        }

    // klastry z dopasowanym sladem (TrackClusterMatch): to nie sa fotony -> pomijane
    IDSet trackMatched;
    for (const auto& name : tcmNames)
      for (const auto& m : frame.get<edm4eic::TrackClusterMatchCollection>(name))
        if (m.getCluster().isAvailable()) markCluster(m.getCluster(), trackMatched);

    // ================= wybor najlepszego kandydata pi0 =================
    PhI bestA, bestB;      // 2 fotony najlepszego kandydata
    int    bestMethod = -1;
    double bestDm = 1e9;
    long   nCand = 0;      // licznik kombinacji w biezacej grupie
    auto consider = [&](const PhI& a, const PhI& b, int method) {
      ++nCand;
      const double dm = std::fabs((a.p4 + b.p4).M() - kMpi0);
      if (dm < bestDm) { bestA = a; bestB = b; bestMethod = method; bestDm = dm; }
    };
    // po kazdej grupie: statystyka (w oknie / good dR<0.2) i informacja, czy mamy pi0 w oknie masy
    auto done = [&](int stage) {
      const TLorentzVector p = bestA.p4 + bestB.p4;
      const bool win = bestMethod >= 0 && p.M() > winLo * kMpi0 && p.M() < winHi * kMpi0;
      if (win) { ++nWin[stage]; if (p.DeltaR(pi0MC) < 0.2) ++nGood[stage]; }
      return win;
    };
    // split klastra na 2 fotony (klasa S); E calkowita klastra dzielona wg energii hitow obu czesci
    auto trySplit = [&](const edm4eic::Cluster& c, double E, int tech, int method) {
      std::vector<Hit> hs, g1, g2;
      IDSet seen;
      gatherHits(c, hs, seen);
      if (!split2(hs, g1, g2, splitSep, splitFrac)) return;
      const Hit c1 = centroid(g1), c2 = centroid(g2);
      consider(PhI{mkPh(c1.pos, E * c1.e / (c1.e + c2.e)), 2, tech},
               PhI{mkPh(c2.pos, E * c2.e / (c1.e + c2.e)), 2, tech}, method);
    };

    // ================= grupa 1: P+P (metoda 0) =================
    nCand = 0;
    for (size_t i = 0; i < gams.size(); ++i)
      for (size_t j = i + 1; j < gams.size(); ++j) {
        potPi0_m->Fill((gams[i].ph.p4 + gams[j].ph.p4).M());
        consider(gams[i].ph, gams[j].ph, 0);
      }
    h_nCand[0]->Fill(nCand);
    bool win = done(0);

    // ================= improved klastry (liczone zawsze - do statystyk) =================
    // improved = klaster nieuzyty przez zadna czastke reko i bez dopasowanego sladu
    std::vector<PhI> improved;                  // klasa I, technologia = kolekcja klastra
    std::vector<edm4eic::Cluster> improvedCl;   // te same klastry jako obiekty (hity potrzebne do splitu)
    {
      IDSet used;
      for (const auto& rp : recos) for (const auto& c : rp.getClusters()) markCluster(c, used);
      int nTrk = 0, nTrue = 0;
      for (size_t ic = 0; ic < clNames.size(); ++ic)   // najpierw scalony Barrel, zeby jego podklastry nie weszly drugi raz
        for (const auto& c : frame.get<edm4eic::ClusterCollection>(clNames[ic])) {
          if (used.count(keyOf(c.getObjectID()))) continue;
          markCluster(c, used);
          if (c.getEnergy() < E_CUT) continue;
          auto it = clInfo.find(keyOf(c.getObjectID()));
          const int cat = it != clInfo.end() ? it->second.cat : 7;
          const int pdg = it != clInfo.end() ? it->second.pdg : 0;
          if (trackMatched.count(keyOf(c.getObjectID()))) {   // ma slad -> pomijamy
            ++nTrk; h_impCatTrk->Fill(cat);
            if (cat != 0) ++pdgImpTrk[pdg];
            continue;
          }
          h_impCat->Fill(cat);
          if (cat == 0) ++nTrue; else ++pdgImp[pdg];
          improved.push_back(PhI{mkPh(TVector3(c.getPosition().x, c.getPosition().y, c.getPosition().z), c.getEnergy()), 1, techOfCl[ic]});
          improvedCl.push_back(c);
        }
      h_nImproved->Fill(improved.size());
      h_nImprovedTrk->Fill(nTrk);
      h_nImpTrue->Fill(nTrue);
    }

    // ================= grupa 2: split P, gdy DOKLADNIE 1 potential gamma (metoda 1) =================
    if (!win && gams.size() == 1) {
      nCand = 0;
      for (const auto& c : gams[0].rec.getClusters())
        if (c.isAvailable()) {
          auto it = clTech.find(keyOf(c.getObjectID()));
          trySplit(c, gams[0].ph.p4.E(), it != clTech.end() ? it->second : -1, 1);
          break;
        }
      h_nCand[1]->Fill(nCand);
    }
    win = done(1);

    // ================= grupa 3: P+I (metoda 2), I+I (metoda 3) =================
    if (!win) {
      nCand = 0;
      for (const auto& g : gams) for (const auto& ip : improved) consider(g.ph, ip, 2);
      for (size_t i = 0; i < improved.size(); ++i)
        for (size_t j = i + 1; j < improved.size(); ++j) consider(improved[i], improved[j], 3);
      h_nCand[2]->Fill(nCand);
    }
    win = done(2);

    // ================= grupa 4: split kazdego improved klastra (metoda 4) =================
    if (!win) {
      nCand = 0;
      for (size_t i = 0; i < improvedCl.size(); ++i)
        trySplit(improvedCl[i], improvedCl[i].getEnergy(), improved[i].tech, 4);
      h_nCand[3]->Fill(nCand);
    }
    win = done(3);

    // ================= grupa 5: P+H (metoda 5), I+H (metoda 6), H+H (metoda 7) =================
    // H = nowe klastry z hitow nieuzytych w zadnym klastrze (takze w klastrach ze sladem); klasa H, technologia = kolekcja hitow
    if (!win) {
      nCand = 0;
      IDSet usedHits;   // hity nalezace do JAKIEGOKOLWIEK klastra
      for (const auto& name : clNames)
        for (const auto& c : frame.get<edm4eic::ClusterCollection>(name))
          for (const auto& h : c.getHits()) if (h.isAvailable()) usedHits.insert(keyOf(h.getObjectID()));

      std::vector<PhI> newCl;
      for (size_t ih = 0; ih < hitNames.size(); ++ih) {
        // seed = najsilniejszy wolny hit, grupa = hity w dR < hitDR; klaster gdy > nHitsMin hitow i E > E_CUT
        std::vector<Hit> hs;
        for (const auto& h : frame.get<edm4eic::CalorimeterHitCollection>(hitNames[ih]))
          if (!usedHits.count(keyOf(h.getObjectID())))
            hs.push_back({TVector3(h.getPosition().x, h.getPosition().y, h.getPosition().z), h.getEnergy()});
        std::sort(hs.begin(), hs.end(), [](const Hit& a, const Hit& b) { return a.e > b.e; });
        std::vector<char> taken(hs.size(), 0);
        for (size_t i = 0; i < hs.size(); ++i) {
          if (taken[i]) continue;
          std::vector<Hit> grp;
          for (size_t j = i; j < hs.size(); ++j)
            if (!taken[j] && hs[j].pos.DeltaR(hs[i].pos) < hitDR) { grp.push_back(hs[j]); taken[j] = 1; }
          const Hit c = centroid(grp);
          if ((int)grp.size() > nHitsMin && c.e > E_CUT) newCl.push_back(PhI{mkPh(c.pos, c.e), 3, techOfHit[ih]});
        }
      }
      h_nNewCl->Fill(newCl.size());

      for (const auto& g : gams)      for (const auto& c : newCl) consider(g.ph, c, 5);
      for (const auto& ip : improved) for (const auto& c : newCl) consider(ip, c, 6);
      for (size_t i = 0; i < newCl.size(); ++i)
        for (size_t j = i + 1; j < newCl.size(); ++j) consider(newCl[i], newCl[j], 7);
      h_nCand[4]->Fill(nCand);
    }
    win = done(4);

    // ================= final pi0 i final fotony (tylko pi0 w oknie masy) =================
    if (!win) continue;
    const TLorentzVector p = bestA.p4 + bestB.p4;
    const int k  = bestMethod;
    const int gp = grpOfMethod[k];
    const double drPi0 = p.DeltaR(pi0MC);

    finalPi0_pt->Fill(p.Pt()); finalPi0_eta->Fill(p.Eta()); finalPi0_E->Fill(p.E()); finalPi0_m->Fill(p.M());
    finalPi0_dR->Fill(drPi0);
    fPi0_E[k]->Fill(p.E()); fPi0_eta[k]->Fill(p.Eta()); fPi0_m[k]->Fill(p.M()); fPi0_dR[k]->Fill(drPi0);
    h_pi_E_res->Fill((pi0MC.E() - p.E()) / pi0MC.E());
    h_pi_eta_res->Fill(pi0MC.Eta() - p.Eta());
    h2_pi0_E[gp]->Fill(pi0MC.E(), p.E());
    h2_pi0_eta[gp]->Fill(pi0MC.Eta(), p.Eta());
    h2_pi0_phi[gp]->Fill(pi0MC.Phi(), p.Phi());
    ++nPi0Grp[gp];
    h_method->Fill(k);
    if (drPi0 < 0.2) h_methodGood->Fill(k);

    const PhI* fin[2] = {&bestA, &bestB};

    // rozdzielczosc fotonow: kazdy foton MC z najblizszym (dR) fotonem final
    for (const auto& mcg : photons) {
      const TLorentzVector& near = (mcg.DeltaR(fin[0]->p4) <= mcg.DeltaR(fin[1]->p4)) ? fin[0]->p4 : fin[1]->p4;
      h_gam_E_res->Fill((mcg.E() - near.E()) / mcg.E());
      h_gam_eta_res->Fill(mcg.Eta() - near.Eta());
    }

    // final fotony: 1D (wszystkie metody razem i osobno dla metody k) oraz 2D MC vs reko (klasa x technologia)
    for (const PhI* g : fin) {
      const TLorentzVector& gp4 = g->p4;
      const double d0 = gp4.DeltaR(photons[0]), d1 = gp4.DeltaR(photons[1]);   // dR do fotonow MC
      finalGamma_pt->Fill(gp4.Pt()); finalGamma_eta->Fill(gp4.Eta()); finalGamma_E->Fill(gp4.E());
      finalGamma_dR->Fill(std::min(d0, d1));
      fGam_E[k]->Fill(gp4.E()); fGam_eta[k]->Fill(gp4.Eta()); fGam_dR[k]->Fill(std::min(d0, d1));

      if (g->tech < 0) { ++nGamNoTech[g->cls]; continue; }
      const TLorentzVector& mcg = (d0 <= d1) ? photons[0] : photons[1];   // foton MC = blizszy, bez ciec
      h2_gam_E[g->cls][g->tech]->Fill(mcg.E(), gp4.E());
      h2_gam_eta[g->cls][g->tech]->Fill(mcg.Eta(), gp4.Eta());
      h2_gam_phi[g->cls][g->tech]->Fill(mcg.Phi(), gp4.Phi());
      ++nGamFill[g->cls][g->tech];
    }
  }

  // PRZED fout->Write(): zapamietaj biny, bo fout->Close() kasuje histogramy
  double methodBins[kNM], methodGoodBins[kNM], impCat[kNC], impCatTrk[kNC], potCat[kNC];
  for (int k = 0; k < kNM; ++k) { methodBins[k] = h_method->GetBinContent(k + 1); methodGoodBins[k] = h_methodGood->GetBinContent(k + 1); }
  for (int i = 0; i < kNC; ++i) {
    impCat[i] = h_impCat->GetBinContent(i + 1); impCatTrk[i] = h_impCatTrk->GetBinContent(i + 1); potCat[i] = h_potCat->GetBinContent(i + 1);
  }

  fout->Write();
  fout->Close();

  // ---------------- podsumowanie ----------------
  std::cout << "\nMC: pi0 z 2 fotonami (E>" << E_CUT << " GeV): " << nPi0 << " (eventow: " << nEv << ")\n";
  std::cout << "zdarzenia z wierzcholkiem z CentralTrackVertices: " << nVtxEv << " / " << nPi0 << "\n";
  std::cout << "pi0 w oknie masy [" << winLo << ", " << winHi << "]*m(pi0); good: dR<0.2\n";
  const char* stageName[kNS] = {"1) P+P", "2) + split P (1 gamma)", "3) + P+I, I+I", "4) + split I", "5) + P+H, I+H, H+H (final)"};
  for (int s = 0; s < kNS; ++s)
    printf("  %-28s w oknie: %6ld (%.1f%%)   good: %6ld (%.1f%%)\n", stageName[s], nWin[s],
           nPi0 ? 100.0 * nWin[s] / nPi0 : 0.0, nGood[s], nPi0 ? 100.0 * nGood[s] / nPi0 : 0.0);
  std::cout << "Metoda finalnego pi0 (w oknie / good):\n";
  for (int k = 0; k < kNM; ++k) printf("  %-20s %6.0f / %6.0f\n", methodLabel[k], methodBins[k], methodGoodBins[k]);

  std::cout << "\nPochodzenie klastrow (truth): improved zostawione / improved odrzucone (slad) / klaster potential gamma\n";
  const char* catPlain[kNC] = {"gamma z pi0", "inny gamma", "e+-", "mu+-", "pi+-", "n/nbar", "inne", "brak asocjacji"};
  for (int i = 0; i < kNC; ++i) printf("  %-16s %8.0f / %8.0f / %8.0f\n", catPlain[i], impCat[i], impCatTrk[i], potCat[i]);
  const std::pair<const char*, const std::map<int, long>*> topPdg[2] = {
      {"Improved zostawione, nie gamma z pi0 - najczestsze PDG czastki MC:", &pdgImp},
      {"Improved odrzucone (slad), nie gamma z pi0 - najczestsze PDG czastki MC:", &pdgImpTrk}};
  for (const auto& tp : topPdg) {
    std::vector<std::pair<long, int>> v;
    for (const auto& kv : *tp.second) v.push_back({kv.second, kv.first});
    std::sort(v.rbegin(), v.rend());
    std::cout << tp.first << "\n";
    for (size_t i = 0; i < v.size() && i < 8; ++i) std::cout << "  PDG " << v[i].second << ": " << v[i].first << "\n";
  }

  std::cout << "\nFotony final pi0 wpisane do histogramow 2D (klasa x technologia):\n";
  printf("  %-6s", "");
  for (int t = 0; t < kNTech; ++t) printf(" %9s", techTag[t]);
  printf(" %12s\n", "bez techn.");
  for (int c = 0; c < kNCls; ++c) {
    printf("  %-6s", clsTag[c]);
    for (int t = 0; t < kNTech; ++t) printf(" %9ld", nGamFill[c][t]);
    printf(" %12ld\n", nGamNoTech[c]);
  }
  std::cout << "Final pi0 w histogramach 2D wg grupy: ";
  for (int g = 0; g < kNS; ++g) std::cout << grpTag[g] << " = " << nPi0Grp[g] << (g + 1 < kNS ? ", " : "\n");
  std::cout << "Zapisano: " << out_file << "\n";
}
