
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

static bool split2(const std::vector<Hit>& hs, std::vector<Hit>& g1, std::vector<Hit>& g2,
                   double minSep, double minFrac) {
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
void anal(const std::string& out_file = "pi0_simple.root",
          double E_CUT = 0.1, long nMax = 10000,           // nMax <= 0 : wszystkie eventy
          double winLo = 0.095, double winHi = 0.167,  // okno masy [GeV]
          double splitSep = 0.16, double splitFrac = 0.01,
          double hitDR = 0.2, int nHitsMin = 4,
          double mergeDR = 0.2,                        // laczenie improved klastrow
          double trkDR = 0.2) {                        // dR klaster - kierunek sladu

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
  // grupy hitow do nowych klastrow: barrel = ScFi + Imaging RAZEM, endcapy osobno
  const std::vector<std::vector<std::string>> hitGroups = {
      {"EcalBarrelScFiRecHits", "EcalBarrelImagingRecHits"}, {"EcalEndcapPRecHits"}, {"EcalEndcapNRecHits"}};

  // ---------------- histogramy ----------------
  TH1D* mcPi0_eta  = new TH1D("mcPi0_eta",  "MC #pi^{0};#eta_{MC};N", 100, -5, 5);
  TH1D* accPi0_eta = new TH1D("accPi0_eta", "accepted #pi^{0};#eta_{MC};N", 100, -5, 5);
  TH1D* mcPi0_E = new TH1D("mcPi0_E",  "MC #pi^{0};E_{MC} [GeV];N", 100, 0, 100);
  TH1D* accPi0_E = new TH1D("accPi0_E", "accepted #pi^{0};E_{MC} [GeV];N", 100, 0, 100);

  TH1D* h_nUnused   = new TH1D("h_nUnused",   "klastry nieuzyte przez ReconstructedParticles;N;events", 21, -0.5, 20.5);
  TH1D* h_nImproved = new TH1D("h_nImproved", "improved (nieuzyte, bez sladu);N;events", 21, -0.5, 20.5);
  TH1D* h_nNewCl    = new TH1D("h_nNewCl",    "nowe klastry z hitow (E > E_{cut});N;events", 51, -0.5, 50.5);
  const char* stageLabel[kNG] = {"1: P+P", "1+2: + split P", "1-3: + P+I, I+I", "1-4: + split I", "1-5: + P+H, I+H, H+H"};
  TH1D *cumEta[kNG], *cumM[kNG];
  for (int s = 0; s < kNG; ++s) {
    cumEta[s] = new TH1D(Form("cum_pi0_eta_%d", s), Form("%s;#eta_{#pi^{0}};N", stageLabel[s]), 100, -5, 5);
    cumM[s]   = new TH1D(Form("cum_pi0_m_%d", s),   Form("%s;m_{#gamma#gamma} [GeV];N", stageLabel[s]), 100, 0, 0.5);
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

  long nEv = 0, nPi0 = 0, nAcc = 0;

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

    // ===== potential gamma (P): neutralne czastki reko jako fotony bezmasowe =====
    struct Gam { Ph ph; edm4eic::ReconstructedParticle rec; };
    std::vector<Gam> gams;
    for (const auto& rp : recos) {
      if (std::fabs(rp.getCharge()) > 0.5) continue;
      TLorentzVector g;
      g.SetXYZM(rp.getMomentum().x, rp.getMomentum().y, rp.getMomentum().z, 0.0);
      if (g.E() <= 0) continue;
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
    auto inWindow = [&]() {
      if (bestDm >= 1e9) return false;
      const double m = (bestA.p4 + bestB.p4).M();
      return m > winLo && m < winHi;
    };
    auto trySplit = [&](const std::vector<edm4eic::Cluster>& cs, double E) {   // klastry -> 2 fotony klasy S
      std::vector<Hit> hs, g1, g2;
      IDSet seen;
      for (const auto& c : cs) gatherHits(c, hs, seen);
      if (!split2(hs, g1, g2, splitSep, splitFrac)) return;
      const Hit c1 = centroid(g1), c2 = centroid(g2);
      consider(Ph{mkPh(c1.pos, E * c1.e / (c1.e + c2.e)), S}, Ph{mkPh(c2.pos, E * c2.e / (c1.e + c2.e)), S});
    };

    int grp = -1;   // grupa, ktora dala pi0 w oknie

    // --- grupa 0: P+P ---
    reset();
    for (size_t i = 0; i < gams.size(); ++i)
      for (size_t j = i + 1; j < gams.size(); ++j) consider(gams[i].ph, gams[j].ph);
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
      for (const auto& g : gams) for (const auto& ip : improved) consider(g.ph, ip);
      for (size_t i = 0; i < improved.size(); ++i)
        for (size_t j = i + 1; j < improved.size(); ++j) consider(improved[i], improved[j]);
      if (inWindow()) grp = 2;
    }

    // --- grupa 3: split I ---
    if (grp < 0) {
      reset();
      for (size_t i = 0; i < improvedCl.size(); ++i) trySplit(improvedCl[i], improved[i].p4.E());
      if (inWindow()) grp = 3;
    }

    // --- grupa 4: P+H, I+H, H+H (nowe klastry z wolnych hitow, nie przy sladzie) ---
    if (grp < 0) {
      reset();
      IDSet usedHits;   // hity nalezace do jakiegokolwiek klastra
      for (const auto& name : clNames)
        for (const auto& c : frame.get<edm4eic::ClusterCollection>(name))
          for (const auto& h : c.getHits()) if (h.isAvailable()) usedHits.insert(keyOf(h.getObjectID()));

      std::vector<Ph> newCl;
      for (const auto& names : hitGroups) {     // barrel: ScFi + Imaging w jednej liscie
        std::vector<Hit> hs;
        for (const auto& name : names)
          for (const auto& h : frame.get<edm4eic::CalorimeterHitCollection>(name))
            if (!usedHits.count(keyOf(h.getObjectID())))
              hs.push_back({TVector3(h.getPosition().x, h.getPosition().y, h.getPosition().z), h.getEnergy()});
        std::sort(hs.begin(), hs.end(), [](const Hit& a, const Hit& b) { return a.e > b.e; });
        std::vector<char> taken(hs.size(), 0);
        for (size_t i = 0; i < hs.size(); ++i) {   // seed = najsilniejszy wolny hit
          if (taken[i]) continue;
          std::vector<Hit> grpHits;
          for (size_t j = i; j < hs.size(); ++j)
            if (!taken[j] && hs[j].pos.DeltaR(hs[i].pos) < hitDR) { grpHits.push_back(hs[j]); taken[j] = 1; }
          const Hit c = centroid(grpHits);
          if ((int)grpHits.size() > nHitsMin && c.e > E_CUT && !nearTrack(c.pos))
            newCl.push_back(Ph{mkPh(c.pos, c.e), H});
        }
      }
      h_nNewCl->Fill(newCl.size());
      for (const auto& g : gams)      for (const auto& c : newCl) consider(g.ph, c);
      for (const auto& ip : improved) for (const auto& c : newCl) consider(ip, c);
      for (size_t i = 0; i < newCl.size(); ++i)
        for (size_t j = i + 1; j < newCl.size(); ++j) consider(newCl[i], newCl[j]);
      if (inWindow()) grp = 4;
    }

    if (grp < 0) continue;   // pi0 nie zrekonstruowane

    // ===== final pi0 =====
    ++nAcc;
    const TLorentzVector p = bestA.p4 + bestB.p4;
    accPi0_eta->Fill(pi0MC.Eta());
    accPi0_E->Fill(pi0MC.E());
    h_grp->Fill(grp);
    if (p.DeltaR(pi0MC) < 0.2) h_grpGood->Fill(grp);
    for (int s = grp; s < kNG; ++s) { cumEta[s]->Fill(p.Eta()); cumM[s]->Fill(p.M()); }

    const bool isSplit = (grp == 1 || grp == 3);
    const bool hasH = (bestA.cls == H || bestB.cls == H);
    const bool hasI = (bestA.cls == I || bestB.cls == I);
    const bool cat[3] = {isSplit, hasH, hasI};  
    for (int c = 0; c < 3; ++c)
      if (cat[c]) { h2E[c]->Fill(pi0MC.E(), p.E()); h2Eta[c]->Fill(pi0MC.Eta(), p.Eta()); }
  }

  fout->Write();
  fout->Close();
  std::cout << "Eventow: " << nEv << ", MC pi0 (2 fotony): " << nPi0 << ", zrekonstruowane: " << nAcc
            << " (" << (nPi0 ? 100.0 * nAcc / nPi0 : 0.0) << "%)\nZapisano: " << out_file << "\n";
}
