// =============================================================================
// anal_simple.cxx - rekonstrukcja pi0 -> 2 gamma (wersja do prezentacji)
//
// Kaskada grup (nastepna tylko gdy poprzednia nie dala pi0 w oknie masy):
//   0: P+P            potential gamma (reko czastki, charge 0)
//   1: split P        split klastra potential gamma (tylko gdy dokladnie 1 P)
//   2: P+I, I+I       improved = klaster nieuzyty przez reko, bez sladu
//   3: split I        split kazdego improved klastra
//   4: P+H, I+H, H+H  nowe klastry z wolnych hitow (barrel: ScFi+Imaging RAZEM)
// Slad: TrackClusterMatch ORAZ prosty match z CentralCKFTracks (dR(kierunek sladu,
// kierunek klastra) < trkDR). Klastry blisko sladu nie sa uzywane jako fotony.
//
// CIECIA ZMNIEJSZAJACE FALSZYWE pi0 (dobrane na probce single photon):
//   splitSig     : split klastra tylko gdy S >= splitSig, gdzie
//                  S = separacja osi / (rms1 + rms2 + 0.005) (grupy 1 i 3)
//   hExclDR      : wolne hity w dR < hExclDR od istniejacego klastra nie tworza nowego klastra H;
//                  pary z H musza tez miec dR(gamma1,gamma2) >= hExclDR
//   hEmin        : minimalna energia klastra H [GeV]
//   hPairMinFrac : pary z H (P+H, I+H, H+H) wymagaja E_min/(E1+E2) >= hPairMinFrac
// =============================================================================
#include <edm4hep/MCParticleCollection.h>
#include <edm4eic/ClusterCollection.h>
#include <edm4eic/CalorimeterHitCollection.h>
#include <edm4eic/TrackClusterMatchCollection.h>
#include <edm4eic/TrackCollection.h>
#include <edm4eic/ReconstructedParticleCollection.h>
#include <edm4eic/VertexCollection.h>
#include <podio/Frame.h>
#include <podio/ROOTReader.h>
#include <podio/ObjectID.h>
#include <TMath.h>
#include <TFile.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TVector3.h>
#include <TLorentzVector.h>
#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <algorithm>
#include <cmath>
#include <TVector2.h>
#include <filesystem>

typedef std::pair<uint32_t, uint32_t> IDKey;
typedef std::set<IDKey> IDSet;
static IDKey keyOf(const podio::ObjectID& id) { return {id.collectionID, id.index}; }

static const double kMpi0 = 0.13498;
static const int kNG = 5;                          // liczba grup / etapow
enum Cls { P = 0, I = 1, S = 2, H = 3 };           // klasa fotonu
struct Ph  { TLorentzVector p4; int cls = -1; };
struct Hit { TVector3 pos; double e; };
struct Imp { TVector3 pos; double E; bool img; std::vector<edm4eic::Cluster> cls; };

// srodek ciezkosci hitow wazony energia, e = suma energii
static Hit centroid(const std::vector<Hit>& g) {
  TVector3 c; double s = 0;
  for (const auto& h : g) { c += h.e * h.pos; s += h.e; }
  if (s > 0) c *= 1.0 / s;
  return {c, s};
}
// klaster + podklastry -> zbior ID
static void markCluster(const edm4eic::Cluster& c, IDSet& used, int depth = 0) {
  if (!c.isAvailable()) return;
  used.insert(keyOf(c.getObjectID()));
  if (depth < 2) for (const auto& sub : c.getClusters()) markCluster(sub, used, depth + 1);
}
// hity klastra i podklastrow (bez duplikatow)
static void gatherHits(const edm4eic::Cluster& c, std::vector<Hit>& out, IDSet& seen, int depth = 0) {
  if (!c.isAvailable()) return;
  for (const auto& h : c.getHits()) {
    if (!h.isAvailable() || !seen.insert(keyOf(h.getObjectID())).second) continue;
    out.push_back({TVector3(h.getPosition().x, h.getPosition().y, h.getPosition().z), h.getEnergy()});
  }
  if (depth < 2) for (const auto& sub : c.getClusters()) gatherHits(sub, out, seen, depth + 1);
}

// dR w (eta,phi) z zawinieciem phi
static double dRep(double e1, double p1, double e2, double p2) {
  return std::hypot(e1 - e2, TVector2::Phi_mpi_pi(p1 - p2));
}
// energy-weighted rms w (eta,phi) wokol punktu (e0,p0)
static double rmsEP(const std::vector<Hit>& g, double e0, double p0) {
  double s = 0, s2 = 0;
  for (const auto& h : g) {
    const double d = dRep(h.pos.Eta(), h.pos.Phi(), e0, p0);
    s += h.e; s2 += h.e * d * d;
  }
  return s > 0 ? std::sqrt(s2 / s) : 0.0;
}
// split na 2 grupy:
//  seed A = hit o najwiekszej energii
//  seed B = hit maksymalizujacy E_hit * dR(A,hit)^2, z dR >= minSep
//  5 deterministycznych iteracji k-means (wazonych energia) w (eta,phi)
//  wymagania koncowe: dR_seed >= minSep, E1,E2 >= minE, min(E1,E2)/(E1+E2) >= minFrac
//  sig = separacja osi / (rms1 + rms2 + 0.005): duze dla dwoch zwartych showerow,
//        male dla jednego showera rozcietego na pol (0.005 = podloga ~ rozmiar komorki)
static bool split2(const std::vector<Hit>& hs, std::vector<Hit>& g1, std::vector<Hit>& g2,
                   double minSep, double minFrac, double minE, double& sig) {
  sig = 0.0;
  if (hs.size() < 2) return false;
  size_t iA = 0;
  for (size_t i = 1; i < hs.size(); ++i) if (hs[i].e > hs[iA].e) iA = i;
  const double eA = hs[iA].pos.Eta(), pA = hs[iA].pos.Phi();

  int iB = -1;
  double best = -1;
  for (size_t k = 0; k < hs.size(); ++k) {
    if (k == iA) continue;
    const double d = dRep(hs[k].pos.Eta(), hs[k].pos.Phi(), eA, pA);
    if (d < minSep) continue;
    const double score = hs[k].e * d * d;
    if (score > best) { best = score; iB = (int)k; }
  }
  if (iB < 0) return false;

  double ea = eA, pa = pA, eb = hs[iB].pos.Eta(), pb = hs[iB].pos.Phi();
  for (int it = 0; it < 5; ++it) {
    g1.clear(); g2.clear();
    for (const auto& h : hs) {
      const double he = h.pos.Eta(), hp = h.pos.Phi();
      (dRep(he, hp, ea, pa) <= dRep(he, hp, eb, pb) ? g1 : g2).push_back(h);
    }
    if (g1.empty() || g2.empty()) return false;
    auto update = [](const std::vector<Hit>& g, double& e0, double& p0) {   // srednia wazona energia w (eta,phi)
      double s = 0, se = 0, sp = 0;
      for (const auto& h : g) {
        s  += h.e;
        se += h.e * (h.pos.Eta() - e0);
        sp += h.e * TVector2::Phi_mpi_pi(h.pos.Phi() - p0);
      }
      if (s > 0) { e0 += se / s; p0 = TVector2::Phi_mpi_pi(p0 + sp / s); }
    };
    update(g1, ea, pa);
    update(g2, eb, pb);
  }

  const double E1 = centroid(g1).e, E2 = centroid(g2).e;
  const double sep = dRep(ea, pa, eb, pb);
  if (sep < minSep) return false;
  if (E1 < minE || E2 < minE) return false;
  if (std::min(E1, E2) / (E1 + E2) < minFrac) return false;
  sig = sep / (rmsEP(g1, ea, pa) + rmsEP(g2, eb, pb) + 0.005);
  return true;
}

// =============================================================================
void anal(
    // --- wejście / wyjście ---
    const std::string& out_file = "pi0_simple.root",
    long nMax = 0,                                            // maks. liczba eventów (0 = wszystkie)

    // --- energia i okno masy ---
    double E_CUT = 0.1,                                       // min. energia klastra / fotonu MC [GeV]
    double winLo = 0.095, double winHi = 0.167,                 // okno masy m(γγ) [GeV]

    // --- ślady i łączenie klastrów (improved) ---
    double trkDR   = 0.2,                                     // klaster blisko śladu nie jest fotonem
    double mergeDR = 0.2,                                     // łączenie nieużytych klastrów w jeden
    double impDR = 0.1,
    double impFrac = 0.1,

    // --- split klastra na 2 fotony (grupy 1 i 3) ---
    double splitSep = 0.010, double splitFrac = 0.05, double splitE = 0.10,   // separacja seedów, min. ułamek energii, min. E podklastra
    double splitSig = 1.2, int splitMinHits = 4,                              // bramka significance, min. hitów w podklastrze

    // --- nowe klastry z wolnych hitów (grupa 4, H) ---
    double hitDR = 0.2, int nHitsMin = 4,                     // promień zbierania hitów, min. liczba hitów
    double hExclDR = 0.1, double hEmin = 0.1,                 // strefa wykluczenia wokół klastrów, min. E klastra H
    double hPairMinFrac = 0.1,                                // min. E_min/(E1+E2) dla par z H

    // --- weto na pary różnych obiektów ---
    double pairMinDR = 0.0, double pairMinFrac = 0.0,         // P+P, P+I, I+I i pary z H: min. dR, min. E_min/(E1+E2)


    // --- truth matching i wydajność ---
    double phMatchDR = 0.1,                                   // dopasowanie fotonu reko do MC
    double pi0DR = 0.1, double pi0dE = 0.10){                  // ciasna wydajność pi0: dR i |dE|/E{                   // min E_min/(E1+E2) dla par z H

  // ---------------- pliki wejsciowe ----------------
  const std::string in_dir = "/run/media/epic/Data/PI0/DVMP_pi0_9x130_26.07";
  const std::string prefix = "DVMP_pi0_9x130";
  std::vector<std::string> in_file;
  if (std::filesystem::exists(in_dir))
    for (const auto& e : std::filesystem::directory_iterator(in_dir)) {
      const std::string fn = e.path().filename().string();
      if (e.is_regular_file() && e.path().extension() == ".root" && fn.rfind(prefix, 0) == 0)
        in_file.push_back(e.path().string());
    }
  std::sort(in_file.begin(), in_file.end());
  if (in_file.empty()) { std::cerr << "Brak plikow w " << in_dir << "\n"; return; }

  podio::ROOTReader reader;
  reader.openFiles(in_file);
  TFile* fout = new TFile(out_file.c_str(), "RECREATE");

  const std::vector<std::string> clNames  = {"EcalBarrelClusters", "EcalBarrelScFiClusters", "EcalBarrelImagingClusters",
                                             "EcalEndcapPClusters", "EcalEndcapNClusters"};
  const std::vector<std::string> tcmNames = {"EcalBarrelTrackClusterMatches", "EcalEndcapPTrackClusterMatches",
                                             "EcalEndcapNTrackClusterMatches"};
  // grupy hitow do nowych klastrow: barrel = tylko scfi RAZEM, endcapy osobno
  const std::vector<std::vector<std::string>> hitGroups = {
      {"EcalBarrelScFiRecHits"}, {"EcalEndcapPRecHits"}, {"EcalEndcapNRecHits"}};

  // ---------------- histogramy ----------------
  TH1D* mcPi0_eta  = new TH1D("mcPi0_eta",  "MC #pi^{0};#eta_{MC};N", 100, -5, 5);
  TH1D* accPi0_eta = new TH1D("accPi0_eta", "accepted #pi^{0};#eta_{MC};N", 100, -5, 5);
  TH1D* mcPi0_E = new TH1D("mcPi0_E",  "MC #pi^{0};E_{MC} [GeV];N", 100, 0, 100);
  TH1D* mcPi0_E_close = new TH1D("mcPi0_E_close",  "MC #pi^{0};E_{MC} [GeV];N", 80, 0, 20);
  TH1D* accPi0_E = new TH1D("accPi0_E", "accepted #pi^{0};E_{MC} [GeV];N", 100, 0, 100);
  TH1D* accPi0_E_close = new TH1D("accPi0_E_close", "accepted #pi^{0};E_{MC} [GeV];N", 80, 0, 20);

  TH1D* h_asym = new TH1D("h_asym", "energy asymmetry;|E_{1}-E_{2}|/(E_{1}+E_{2});N", 50, 0, 1);

  TH1D* h_nUnused   = new TH1D("h_nUnused",   "klastry nieuzyte przez ReconstructedParticles;N;events", 21, -0.5, 20.5);
  TH1D* h_nImproved = new TH1D("h_nImproved", "improved (nieuzyte, bez sladu);N;events", 21, -0.5, 20.5);
  TH1D* h_nNewCl    = new TH1D("h_nNewCl",    "nowe klastry z hitow (E > E_{cut});N;events", 51, -0.5, 50.5);
  const char* stageLabel[kNG] = {"1: P+P", "1+2: + split P", "1-3: + P+I, I+I", "1-4: + split I", "1-5: + P+H, I+H, H+H"};
  TH1D *cumEta[kNG], *cumM[kNG], *cumE[kNG], *cumE_close[kNG];
  for (int s = 0; s < kNG; ++s) {
    cumEta[s] = new TH1D(Form("cum_pi0_eta_%d", s), Form("%s;#eta_{#pi^{0}};N", stageLabel[s]), 100, -5, 5);
    cumM[s]   = new TH1D(Form("cum_pi0_m_%d", s),   Form("%s;m_{#gamma#gamma} [GeV];N", stageLabel[s]), 100, 0, 0.5);
    cumE[s]   = new TH1D(Form("cum_pi0_E_%d", s),   Form("%s;E_{#pi^{0}} [GeV];N", stageLabel[s]), 100, 0, 100);
    cumE_close[s] = new TH1D(Form("cum_pi0_E_close_%d", s), Form("%s;E_{#pi^{0}} [GeV];N", stageLabel[s]), 80, 0, 20);
  }
  const char* grpLabel[kNG] = {"P+P", "split P", "P+I, I+I", "split I", "P+H, I+H, H+H"};
  TH1D* h_grp     = new TH1D("h_grp",     "#pi^{0} in mass window;;#pi^{0}", kNG, 0, kNG);
  TH1D* h_grpGood = new TH1D("h_grpGood", "#pi^{0} good (#DeltaR<0.2);;#pi^{0}", kNG, 0, kNG);
  for (int g = 0; g < kNG; ++g) {
    h_grp->GetXaxis()->SetBinLabel(g + 1, grpLabel[g]);
    h_grpGood->GetXaxis()->SetBinLabel(g + 1, grpLabel[g]);
  }
  const char* catTag[3]   = {"split", "hits", "improved"};
  const char* catLabel[3] = {"split (split P, split I)", "from hits (#geq 1 #gamma from hits)", "improved (#geq 1 #gamma improved)"};
  TH2D *h2E[3], *h2Eta[3];
  for (int c = 0; c < 3; ++c) {
    h2E[c]   = new TH2D(Form("h2_pi0_E_%s", catTag[c]),   Form("%s;E_{MC} [GeV];E_{reco} [GeV]", catLabel[c]), 50, 0, 40, 50, 0, 40);
    h2Eta[c] = new TH2D(Form("h2_pi0_eta_%s", catTag[c]), Form("%s;#eta_{MC};#eta_{reco}", catLabel[c]), 100, -5, 5, 100, -5, 5);
  }

  TH1D* tightPi0_eta = new TH1D("tightPi0_eta", "tight #pi^{0};#eta_{MC};N", 100, -5, 5);
  TH1D* tightPi0_E   = new TH1D("tightPi0_E",   "tight #pi^{0};E_{MC} [GeV];N", 100, 0, 100);
  TH1D* matchPi0_eta = new TH1D("matchPi0_eta", "both #gamma matched;#eta_{MC};N", 100, -5, 5);
  TH1D* matchPi0_E   = new TH1D("matchPi0_E",   "both #gamma matched;E_{MC} [GeV];N", 100, 0, 100);

  TH2D* h2_gam_E   = new TH2D("h2_gam_E",   "#gamma (1:1 assignment);E_{MC} [GeV];E_{reco} [GeV]", 50, 0, 40, 50, 0, 40);
  TH2D* h2_gam_eta = new TH2D("h2_gam_eta", "#gamma (1:1 assignment);#eta_{MC};#eta_{reco}", 100, -5, 5, 100, -5, 5);
  TH2D* h2_gam_phi = new TH2D("h2_gam_phi", "#gamma (1:1 assignment);#phi_{MC};#phi_{reco}", 100, -TMath::Pi(), TMath::Pi(), 100, -TMath::Pi(), TMath::Pi());
  TH1D* h_gamCls      = new TH1D("h_gamCls",      "#gamma in mass window by class;;N", 4, 0, 4);
  TH1D* h_gamClsMatch = new TH1D("h_gamClsMatch", "#gamma matched (#DeltaR<cut) by class;;N", 4, 0, 4);
  TH1D* h_pi0_dr = new TH1D("h_pi0_dr", ";#DeltaR;N", 50, 0, 1);
  const char* clsName[4] = {"P", "I", "S", "H"};
  for (int c = 0; c < 4; ++c) {
    h_gamCls->GetXaxis()->SetBinLabel(c + 1, clsName[c]);
    h_gamClsMatch->GetXaxis()->SetBinLabel(c + 1, clsName[c]);
  }

  long nEv = 0, nPi0 = 0, nAcc = 0, nTight = 0, nMatch = 0, nPh = 0, nPhMatch = 0;
  long ngood = 0, npi0rec=0;

  while (auto entry = reader.readNextEntry(podio::Category::Event)) {
    ++nEv;
    if (nMax > 0 && nEv > nMax) break;
    if (nEv % 1000 == 0) std::cout << "Processed " << nEv << " events.\n";
    auto frame = podio::Frame(std::move(entry));
    const auto& mc    = frame.get<edm4hep::MCParticleCollection>("MCParticles");
    const auto& recos = frame.get<edm4eic::ReconstructedParticleCollection>("ReconstructedParticles");

    // ===== MC: pi0 i dokladnie 2 fotony z pi0 (E > E_CUT, poza szczelina 1.3<|eta|<1.6) =====
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
    mcPi0_eta->Fill(pi0MC.Eta());
    mcPi0_E->Fill(pi0MC.E());
    mcPi0_E_close->Fill(pi0MC.E());

    // ===== potential gamma (P): neutralne czastki reko jako fotony bezmasowe =====
    struct Gam { Ph ph; edm4eic::ReconstructedParticle rec; };
    std::vector<Gam> gams;
    for (const auto& rp : recos) {
      if (std::fabs(rp.getCharge()) > 0.5) continue;
      TLorentzVector g;
      g.SetXYZM(rp.getMomentum().x, rp.getMomentum().y, rp.getMomentum().z, 0.0);
      if (g.E() <= 0) continue;
      if (1.3 < std::fabs(g.Eta()) && std::fabs(g.Eta()) < 1.6) continue;
      gams.push_back({Ph{g, P}, rp});
    }

    // ===== wierzcholek: CentralTrackVertices (najwiecej czastek), brak -> (0,0,0) =====
    TVector3 vtx(0, 0, 0);
    {
      long bestN = -1;
      for (const auto& v : frame.get<edm4eic::VertexCollection>("CentralTrackVertices")) {
        const long n = v.getAssociatedParticles().size();
        if (n > bestN) { bestN = n; vtx.SetXYZ(v.getPosition().x, v.getPosition().y, v.getPosition().z); }
      }
    }
    auto mkPh = [&](const TVector3& pos, double E) {   // foton o energii E i kierunku (pos - vtx)
      const TVector3 u = (pos - vtx).Unit();
      return TLorentzVector(E * u.X(), E * u.Y(), E * u.Z(), E);
    };

    // ===== slady: kierunki pedow z CentralCKFTracks =====
    std::vector<TVector3> trkDir;
    for (const auto& t : frame.get<edm4eic::TrackCollection>("CentralCKFTracks")) {
      const TVector3 mom(t.getMomentum().x, t.getMomentum().y, t.getMomentum().z);
      if (mom.Mag() > 0) trkDir.push_back(mom);
    }
    // czy klaster na pozycji pos lezy w dR < trkDR od kierunku jakiegos sladu
    auto nearTrack = [&](const TVector3& pos) {
      const TVector3 d = pos - vtx;
      for (const auto& td : trkDir) if (d.DeltaR(td) < trkDR) return true;
      return false;
    };

    // ===== klastry ze sladem (wbudowany TrackClusterMatch) =====
    IDSet trackMatched;
    for (const auto& name : tcmNames)
      for (const auto& m : frame.get<edm4eic::TrackClusterMatchCollection>(name))
        if (m.getCluster().isAvailable()) markCluster(m.getCluster(), trackMatched);

    // ===== improved: nieuzyte przez reko, bez sladu (TCM i CKF), polaczone w dR < mergeDR =====
    std::vector<Ph> improved;
    std::vector<std::vector<edm4eic::Cluster>> improvedCl;   // klastry skladowe kazdego improved (do splitu)
    {
      IDSet used;
      for (const auto& rp : recos) for (const auto& c : rp.getClusters()) markCluster(c, used);
      int nUnused = 0;
      std::vector<Imp> cand;
      for (size_t ic = 0; ic < clNames.size(); ++ic)   // Barrel pierwszy: jego podklastry nie wejda drugi raz
        for (const auto& c : frame.get<edm4eic::ClusterCollection>(clNames[ic])) {
          if (used.count(keyOf(c.getObjectID()))) continue;
          markCluster(c, used);
          if (c.getEnergy() < E_CUT) continue;
          ++nUnused;
          const TVector3 cpos(c.getPosition().x, c.getPosition().y, c.getPosition().z);
          if (trackMatched.count(keyOf(c.getObjectID())) || nearTrack(cpos)) continue;   // slad -> nie foton
          cand.push_back({cpos, (double)c.getEnergy(), ic == 2, {c}});   // ic == 2 -> EcalBarrelImagingClusters
        }
      h_nUnused->Fill(nUnused);

      // laczenie: E = suma, pozycja = imaging (jesli jest w grupie)
      std::sort(cand.begin(), cand.end(), [](const Imp& a, const Imp& b) { return a.E > b.E; });
      std::vector<char> taken(cand.size(), 0);
      for (size_t i = 0; i < cand.size(); ++i) {
        if (taken[i]) continue;
        taken[i] = 1;
        Imp m = cand[i];                                   // seed = najsilniejszy
        for (size_t j = i + 1; j < cand.size(); ++j) {
          if (taken[j] || cand[j].pos.DeltaR(cand[i].pos) >= mergeDR) continue;
          taken[j] = 1;
          m.E += cand[j].E;
          m.cls.push_back(cand[j].cls[0]);
          if (cand[j].img && !m.img) { m.pos = cand[j].pos; m.img = true; }
        }
        improved.push_back(Ph{mkPh(m.pos, m.E), I});
        improvedCl.push_back(m.cls);
      }
      h_nImproved->Fill(improved.size());
    }

    // ===== wybor najlepszej pary w grupie =====
    Ph bestA, bestB;
    double bestDm = 1e9;
    auto reset = [&]() { bestA = Ph(); bestB = Ph(); bestDm = 1e9; };
    auto consider = [&](const Ph& a, const Ph& b) {
      const double dm = std::fabs((a.p4 + b.p4).M() - kMpi0);
      if (dm < bestDm) { bestA = a; bestB = b; bestDm = dm; }
    };

    auto considerPair = [&](const Ph& a, const Ph& b) {      // weto na fragmenty jednego showera
      if (a.p4.DeltaR(b.p4) < pairMinDR) return;
      if (std::min(a.p4.E(), b.p4.E()) / (a.p4.E() + b.p4.E()) < pairMinFrac) return;
      consider(a, b);
    };
    auto considerPairI = [&](const Ph& a, const Ph& b) {     // pary z improved: dodatkowe cięcia
      if (a.p4.DeltaR(b.p4) < impDR) return;
      if (std::min(a.p4.E(), b.p4.E()) / (a.p4.E() + b.p4.E()) < impFrac) return;
      if (1.3 < std::fabs(a.p4.Eta()) && std::fabs(a.p4.Eta()) < 1.6) return;
      if (1.3 < std::fabs(b.p4.Eta()) && std::fabs(b.p4.Eta()) < 1.6) return;
      considerPair(a, b);
    };

    auto inWindow = [&]() {
      if (bestDm >= 1e9) return false;
      const double m = (bestA.p4 + bestB.p4).M();
      return m > winLo && m < winHi;
    };
    auto trySplit = [&](const std::vector<edm4eic::Cluster>& cs, double E) {   // klastry -> 2 fotony klasy S
      std::vector<Hit> hs, g1, g2;
      IDSet seen;
      for (const auto& c : cs) gatherHits(c, hs, seen);
      double sig = 0;
      if (!split2(hs, g1, g2, splitSep, splitFrac, splitE, sig)) return;
      if (!split2(hs, g1, g2, splitSep, splitFrac, splitE, sig)) return;
      if ((int)g1.size() < splitMinHits || (int)g2.size() < splitMinHits) return;
      if (sig < splitSig) return;                                              // bramka: wyglada na jeden shower
      const Hit c1 = centroid(g1), c2 = centroid(g2);
      consider(Ph{mkPh(c1.pos, E * c1.e / (c1.e + c2.e)), S}, Ph{mkPh(c2.pos, E * c2.e / (c1.e + c2.e)), S});
    };

    int grp = -1;   // grupa, ktora dala pi0 w oknie

    // --- grupa 0: P+P ---
    reset();
    for (size_t i = 0; i < gams.size(); ++i)
      for (size_t j = i + 1; j < gams.size(); ++j) considerPair(gams[i].ph, gams[j].ph);
    if (inWindow()) grp = 0;

    // --- grupa 1: split P (dokladnie 1 potential gamma) ---
    if (grp < 0 && gams.size() == 1) {
      reset();
      for (const auto& c : gams[0].rec.getClusters())
        if (c.isAvailable()) { trySplit({c}, gams[0].ph.p4.E()); break; }
      if (inWindow()) grp = 1;
    }

    // --- grupa 2: P+I, I+I ---
    if (grp < 0) {
      reset();
      for (const auto& g : gams) for (const auto& ip : improved) considerPairI(g.ph, ip);
      for (size_t i = 0; i < improved.size(); ++i)
        for (size_t j = i + 1; j < improved.size(); ++j) considerPairI(improved[i], improved[j]);
      if (inWindow()) grp = 2;
    }

    // --- grupa 3: split I ---
    if (grp < 0) {
      reset();
      for (size_t i = 0; i < improvedCl.size(); ++i) {
        if(1.3 < std::fabs(improved[i].p4.Eta()) && std::fabs(improved[i].p4.Eta()) < 1.6) continue;
        trySplit(improvedCl[i], improved[i].p4.E());
      }
      if (inWindow()) grp = 3;
    }

    // --- grupa 4: P+H, I+H, H+H (nowe klastry z wolnych hitow, nie przy sladzie) ---
    if (grp < 0) {
      reset();
      IDSet usedHits;                  // hity nalezace do jakiegokolwiek klastra
      std::vector<TVector3> exclPos;   // kierunki istniejacych klastrow (strefa wykluczenia)
      for (const auto& name : clNames)
        for (const auto& c : frame.get<edm4eic::ClusterCollection>(name)) {
          for (const auto& h : c.getHits()) if (h.isAvailable()) usedHits.insert(keyOf(h.getObjectID()));
          if (c.getEnergy() >= E_CUT)
            exclPos.push_back(TVector3(c.getPosition().x, c.getPosition().y, c.getPosition().z) - vtx);
        }
      auto nearCluster = [&](const TVector3& hp) {
        const TVector3 d = hp - vtx;
        for (const auto& cp : exclPos) if (d.DeltaR(cp) < hExclDR) return true;
        return false;
      };
      auto considerH = [&](const Ph& a, const Ph& b) {   // pary z H: asymetria i minimalny kat
        if (std::min(a.p4.E(), b.p4.E()) / (a.p4.E() + b.p4.E()) < hPairMinFrac) return;
        if (1.3 < std::fabs(a.p4.Eta()) && std::fabs(a.p4.Eta()) < 1.6) return;
        if (1.3 < std::fabs(b.p4.Eta()) && std::fabs(b.p4.Eta()) < 1.6) return;
        if (a.p4.DeltaR(b.p4) < hExclDR) return;
        considerPair(a, b);
      };
     

      std::vector<Ph> newCl;
      for (const auto& names : hitGroups) {     // barrel: tylko scifi
        std::vector<Hit> hs;
        for (const auto& name : names)
          for (const auto& h : frame.get<edm4eic::CalorimeterHitCollection>(name)) {
            if (usedHits.count(keyOf(h.getObjectID()))) continue;
            const TVector3 hp(h.getPosition().x, h.getPosition().y, h.getPosition().z);
            if (hExclDR > 0 && nearCluster(hp)) continue;   // halo istniejacego klastra
            hs.push_back({hp, h.getEnergy()});
          }
        std::sort(hs.begin(), hs.end(), [](const Hit& a, const Hit& b) { return a.e > b.e; });
        std::vector<char> taken(hs.size(), 0);
        for (size_t i = 0; i < hs.size(); ++i) {   // seed = najsilniejszy wolny hit
          if (taken[i]) continue;
          std::vector<Hit> grpHits;
          for (size_t j = i; j < hs.size(); ++j)
            if (!taken[j] && hs[j].pos.DeltaR(hs[i].pos) < hitDR) { grpHits.push_back(hs[j]); taken[j] = 1; }
          const Hit c = centroid(grpHits);
         
          if ((int)grpHits.size() <= nHitsMin || c.e <= hEmin || nearTrack(c.pos)) continue;
            newCl.push_back(Ph{mkPh(c.pos, c.e), H});
        }
      }
      h_nNewCl->Fill(newCl.size());
      for (const auto& g : gams)      for (const auto& c : newCl) considerH(g.ph, c);
      for (const auto& ip : improved) for (const auto& c : newCl) considerH(ip, c);
      for (size_t i = 0; i < newCl.size(); ++i)
        for (size_t j = i + 1; j < newCl.size(); ++j) considerH(newCl[i], newCl[j]);
      if (inWindow()) grp = 4;
    }

    if (grp < 0) continue;   // pi0 nie zrekonstruowane

    // ===== final pi0 =====
    ++nAcc;
    const TLorentzVector p = bestA.p4 + bestB.p4;
    accPi0_eta->Fill(pi0MC.Eta());
    accPi0_E->Fill(pi0MC.E());
    accPi0_E_close->Fill(pi0MC.E());
    h_grp->Fill(grp);
    h_pi0_dr->Fill(p.DeltaR(pi0MC));

    // ===== truth matching fotonow 1:1 (lepsza z dwoch permutacji) =====
    const Ph* rph[2] = {&bestA, &bestB};
    const double d00 = rph[0]->p4.DeltaR(photons[0]) + rph[1]->p4.DeltaR(photons[1]);
    const double d01 = rph[0]->p4.DeltaR(photons[1]) + rph[1]->p4.DeltaR(photons[0]);
    const int mt[2] = {d00 <= d01 ? 0 : 1, d00 <= d01 ? 1 : 0};   // foton reko k -> foton MC mt[k]
    int nGood = 0;
    for (int k = 0; k < 2; ++k) {
      const TLorentzVector& t = photons[mt[k]];
      const TLorentzVector& r = rph[k]->p4;
      ++nPh;
      h_gamCls->Fill(rph[k]->cls);
      h2_gam_E->Fill(t.E(), r.E());
      h2_gam_eta->Fill(t.Eta(), r.Eta());
      h2_gam_phi->Fill(t.Phi(), r.Phi());
      if (r.DeltaR(t) < phMatchDR) { ++nGood; ++nPhMatch; h_gamClsMatch->Fill(rph[k]->cls); }
    }
    if (nGood == 2) { ++nMatch; matchPi0_eta->Fill(pi0MC.Eta()); matchPi0_E->Fill(pi0MC.E()); }

    // ===== druga wydajnosc pi0: ciasne cuty na pi0 =====
    if (p.DeltaR(pi0MC) < pi0DR && std::fabs(pi0MC.E() - p.E()) / pi0MC.E() < pi0dE) {
      ++nTight;
      tightPi0_eta->Fill(pi0MC.Eta());
      tightPi0_E->Fill(pi0MC.E());
    }
    if (p.DeltaR(pi0MC) < 0.1) {h_grpGood->Fill(grp); ++ngood;}
    for (int s = grp; s < kNG; ++s) { cumEta[s]->Fill(p.Eta()); cumM[s]->Fill(p.M()); cumE[s]->Fill(p.E()); cumE_close[s]->Fill(p.E()); }

    const bool isSplit = (grp == 1 || grp == 3);
    const bool hasH = (bestA.cls == H || bestB.cls == H);
    const bool hasI = (bestA.cls == I || bestB.cls == I);
    const bool cat[3] = {isSplit, hasH, hasI};   // kategorie moga sie nakladac (np. I+H)
    for (int c = 0; c < 3; ++c)
      if (cat[c]) { h2E[c]->Fill(pi0MC.E(), p.E()); h2Eta[c]->Fill(pi0MC.Eta(), p.Eta()); }
  }

  fout->Write();
  fout->Close();
  auto pct = [&](long n, long d) { return d ? 100.0 * n / d : 0.0; };
  std::cout << "Eventow: " << nEv << ", MC pi0 (2 fotony): " << nPi0 << "\n"
            << "wydajnosc (okno masy):          " << nAcc   << " (" << pct(nAcc, nPi0)   << "%)\n"
            << "wydajnosc ciasna (dR<" << pi0DR << ", |dE|/E<" << pi0dE << "): " << nTight << " (" << pct(nTight, nPi0) << "%)\n"
            << "oba fotony dopasowane (dR<" << phMatchDR << "): " << nMatch << " (" << pct(nMatch, nPi0) << "%)\n"
            << "fotony dopasowane: " << nPhMatch << "/" << nPh << " (" << pct(nPhMatch, nPh) << "% w eventach z oknem)\n"
            << "pi0 dobrze dopasowane (dR<0.1): " << ngood << " (" << pct(ngood, nPi0) << "%) wszystkich pi0\n"
            << "pi0 good (dR<0.1)/accepted: " << ngood << " (" << pct(ngood, nAcc) << "%) sposrod zaakceptowanych\n"
            << "Zapisano: " << out_file << "\n";
}
