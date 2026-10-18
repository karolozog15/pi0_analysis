
#include <edm4hep/MCParticleCollection.h>
#include <edm4eic/ClusterCollection.h>
#include <edm4eic/CalorimeterHitCollection.h>
#include <edm4eic/MCRecoParticleAssociationCollection.h>
#include <edm4eic/MCRecoClusterParticleAssociationCollection.h>
#include <edm4eic/TrackClusterMatchCollection.h>
#include <edm4eic/ReconstructedParticleCollection.h>

#include <podio/Frame.h>
#include <podio/ROOTReader.h>
#include <podio/ObjectID.h>

#include <TFile.h>
#include <TCanvas.h>
#include <TH1D.h>
#include <TH3F.h>
#include <TPolyMarker3D.h>
#include <TPolyLine3D.h>
#include <TLegend.h>
#include <TColor.h>
#include <TROOT.h>
#include <TVector3.h>

#include <filesystem>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <cmath>
#include <algorithm>

using ID = std::pair<uint32_t, uint32_t>;
static ID idOf(const podio::ObjectID& o) { return {o.collectionID, o.index}; }

// ---------------------------------------------------------------- drawing structures
struct Hit   { double x, y, z, E; };
struct Group { std::string label; int family; std::vector<Hit> hits; }; // family: 0=ScFi(blue) 1=Imaging(red) 2=Endcap(green)
struct McLine { TVector3 v, e, mom; };                                  // getVertex, getEndpoint, MC photon momentum
struct Extras {
  std::vector<McLine> mc;
  std::vector<std::pair<TVector3, std::string>> stars;                  // cluster.position
};

static double gViewTheta = 20, gViewPhi = -20, gMarkerSize = 3.0;
static const int NL = 6; // number of energy levels in the gradient
static int colIdx(int family, int lvl) {
  static const int scfi[NL] = {kAzure + 7, kAzure + 1, kBlue,     kBlue + 1,   kBlue + 2,   kBlue + 3};
  static const int img[NL]  = {kOrange + 1, kOrange + 7, kRed - 4, kRed,        kRed + 1,    kRed + 3};
  static const int ecp[NL]  = {kSpring + 7, kSpring + 5, kGreen,  kGreen + 1,  kGreen + 2,  kGreen + 3};
  return family == 0 ? scfi[lvl] : family == 1 ? img[lvl] : ecp[lvl];
}

static void initColors() {
  // strong, saturated colors: low E -> high E
  // ScFi: light blue -> navy | Imaging: orange -> dark red | Endcap: green -> dark green
  const double lo[3][3] = {{0.0, 0.55, 1.0}, {1.0, 0.55, 0.0}, {0.3, 0.9, 0.0}};
  const double hi[3][3] = {{0.0, 0.0, 0.6},  {0.7, 0.0, 0.0},  {0.0, 0.4, 0.0}};
  for (int f = 0; f < 3; ++f)
    for (int l = 0; l < NL; ++l) {
      const double t = double(l) / (NL - 1);
      new TColor(colIdx(f, l), lo[f][0] + (hi[f][0] - lo[f][0]) * t,
                               lo[f][1] + (hi[f][1] - lo[f][1]) * t,
                               lo[f][2] + (hi[f][2] - lo[f][2]) * t);
    }
}

// ---------------------------------------------------------------- clusters / hits (as in anal_res.cxx)
static void markCluster(const edm4eic::Cluster& c, std::set<ID>& used, int depth = 0) {
  if (!c.isAvailable()) return;
  used.insert(idOf(c.getObjectID()));
  if (depth < 2) for (const auto& sub : c.getClusters()) markCluster(sub, used, depth + 1);
}
static void gatherHits(const edm4eic::Cluster& c, std::vector<Hit>& out, std::set<ID>& seen, int depth = 0) {
  if (!c.isAvailable()) return;
  for (const auto& h : c.getHits()) {
    if (!h.isAvailable() || !seen.insert(idOf(h.getObjectID())).second) continue;
    out.push_back({h.getPosition().x, h.getPosition().y, h.getPosition().z, h.getEnergy()});
  }
  if (depth < 2) for (const auto& sub : c.getClusters()) gatherHits(sub, out, seen, depth + 1);
}
static TVector3 posOf(const edm4eic::Cluster& c) {
  return TVector3(c.getPosition().x, c.getPosition().y, c.getPosition().z);
}

// ---------------------------------------------------------------- clipping a line to a box
// line a + t(b-a) clipped to [lo,hi]; infinite=false -> t in [0,1]
static bool clipLine(const double a[3], const double b[3], const double lo[3], const double hi[3],
                     bool infinite, double& t0, double& t1) {
  t0 = infinite ? -1e9 : 0; t1 = infinite ? 1e9 : 1;
  for (int k = 0; k < 3; ++k) {
    const double d = b[k] - a[k];
    if (std::fabs(d) < 1e-12) { if (a[k] < lo[k] || a[k] > hi[k]) return false; continue; }
    double ta = (lo[k] - a[k]) / d, tb = (hi[k] - a[k]) / d;
    if (ta > tb) std::swap(ta, tb);
    t0 = std::max(t0, ta); t1 = std::min(t1, tb);
    if (t0 > t1) return false;
  }
  return true;
}

// ---------------------------------------------------------------- 3D drawing
static bool drawExample(TFile* fout, const std::string& name, const std::string& title,
                        const std::vector<Group>& groups, bool savePng, const Extras& ex) {
  double lo[3] = {1e18, 1e18, 1e18}, hi[3] = {-1e18, -1e18, -1e18}, Emax = 0;
  size_t nTot = 0;
  auto ext = [&](double z, double x, double y) {
    const double v[3] = {z, x, y}; // [0]=z (beam axis, horizontal), [1]=x, [2]=y (vertical)
    for (int k = 0; k < 3; ++k) { lo[k] = std::min(lo[k], v[k]); hi[k] = std::max(hi[k], v[k]); }
  };
  for (auto& g : groups) for (auto& h : g.hits) { ext(h.z, h.x, h.y); Emax = std::max(Emax, h.E); ++nTot; }
  if (nTot == 0) return false;
  for (auto& s : ex.stars) ext(s.first.Z(), s.first.X(), s.first.Y());
  for (int k = 0; k < 3; ++k) { double p = std::max(5.0, 0.1 * (hi[k] - lo[k])); lo[k] -= p; hi[k] += p; }

  TCanvas* cv = new TCanvas(name.c_str(), title.c_str(), 1200, 1000);
  cv->SetTheta(gViewTheta); cv->SetPhi(gViewPhi);
  TH3F* fr = new TH3F((name + "_frame").c_str(), (title + ";z [mm];x [mm];y [mm]").c_str(),
                      1, lo[0], hi[0], 1, lo[1], hi[1], 1, lo[2], hi[2]);
  fr->SetDirectory(nullptr); fr->SetStats(0); fr->Draw();
  TLegend* leg = new TLegend(0.40, 0.78, 0.99, 0.99);

  for (auto& g : groups) {
    std::vector<std::vector<Hit>> byLvl(NL);
    double eSum = 0;
    for (auto& h : g.hits) {
      const double f = Emax > 0 ? std::sqrt(h.E / Emax) : 0;
      byLvl[std::min(NL - 1, int(f * NL))].push_back(h);
      eSum += h.E;
    }
    TPolyMarker3D* lastPm = nullptr;
    for (int l = 0; l < NL; ++l) {
      if (byLvl[l].empty()) continue;
      auto* pm = new TPolyMarker3D(byLvl[l].size());
      for (size_t i = 0; i < byLvl[l].size(); ++i) pm->SetPoint(i, byLvl[l][i].z, byLvl[l][i].x, byLvl[l][i].y);
      pm->SetMarkerStyle(20); pm->SetMarkerSize(gMarkerSize); pm->SetMarkerColor(colIdx(g.family, l));
      pm->Draw(); lastPm = pm;
    }
    if (lastPm) leg->AddEntry(lastPm, Form("%s: %zu hit, #SigmaE=%.3f GeV", g.label.c_str(), g.hits.size(), eSum), "p");
  }

  // MC lines: solid = vertex->endpoint (within the hit range), dashed = extension
  bool legSeg = false, legExt = false;
  for (const auto& m : ex.mc) {
    const double segLen = (m.e - m.v).Mag();
    const bool hasSeg = segLen > 1e-6;
    const TVector3 dir = hasSeg ? (m.e - m.v) : m.mom;   // vertex==endpoint -> use momentum
    const double a[3] = {m.v.Z(), m.v.X(), m.v.Y()};
    const double b[3] = {m.v.Z() + dir.Z(), m.v.X() + dir.X(), m.v.Y() + dir.Y()};
    double t0, t1;
    if (clipLine(a, b, lo, hi, true, t0, t1)) {
      auto* ln = new TPolyLine3D(2);
      ln->SetPoint(0, a[0] + t0 * (b[0] - a[0]), a[1] + t0 * (b[1] - a[1]), a[2] + t0 * (b[2] - a[2]));
      ln->SetPoint(1, a[0] + t1 * (b[0] - a[0]), a[1] + t1 * (b[1] - a[1]), a[2] + t1 * (b[2] - a[2]));
      ln->SetLineColor(kMagenta - 9); ln->SetLineWidth(2); ln->SetLineStyle(2);
      ln->Draw();
      if (!legExt) { leg->AddEntry(ln, "MC line extended", "l"); legExt = true; }
    }
    if (hasSeg && clipLine(a, b, lo, hi, false, t0, t1)) {
      auto* ln = new TPolyLine3D(2);
      ln->SetPoint(0, a[0] + t0 * (b[0] - a[0]), a[1] + t0 * (b[1] - a[1]), a[2] + t0 * (b[2] - a[2]));
      ln->SetPoint(1, a[0] + t1 * (b[0] - a[0]), a[1] + t1 * (b[1] - a[1]), a[2] + t1 * (b[2] - a[2]));
      ln->SetLineColor(kMagenta + 1); ln->SetLineWidth(5); ln->SetLineStyle(1);
      ln->Draw();
      if (!legSeg) { leg->AddEntry(ln, "MC vertex->endpoint", "l"); legSeg = true; }
    }
  }
  if (!legSeg) leg->AddEntry((TObject*)nullptr, "MC endpoint nie dochodzi do hitow", "");

  // cluster.position as a star
  for (auto& s : ex.stars) {
    auto* st = new TPolyMarker3D(1);
    st->SetPoint(0, s.first.Z(), s.first.X(), s.first.Y());
    st->SetMarkerStyle(29); st->SetMarkerSize(4); st->SetMarkerColor(kBlack);
    st->Draw();
    leg->AddEntry(st, Form("cluster.position (%s)", s.second.c_str()), "p");
  }
  leg->Draw();

  fout->cd();
  cv->Write(name.c_str());
  if (savePng) cv->SaveAs((name + ".png").c_str());
  delete cv;
  return true;
}

// ---------------------------------------------------------------- input files
static std::vector<std::string> collectInputFiles(const std::string& dir, const std::string& prefix) {
  std::vector<std::string> files;
  if (!std::filesystem::exists(dir)) return files;
  for (const auto& entry : std::filesystem::directory_iterator(dir)) {
    if (!entry.is_regular_file()) continue;
    const std::string filename = entry.path().filename().string();
    if (entry.path().extension() != ".root") continue;
    if (filename.rfind(prefix, 0) != 0) continue;
    files.push_back(entry.path().string());
  }
  std::sort(files.begin(), files.end());
  return files;
}

static double dRpos(const TVector3& a, const TVector3& b) {
  double dp = a.Phi() - b.Phi();
  while (dp > M_PI)  dp -= 2 * M_PI;
  while (dp < -M_PI) dp += 2 * M_PI;
  return std::hypot(a.Eta() - b.Eta(), dp);
}

// ---------------------------------------------------------------- main function
void hit3d(const std::string& out_file = "hits3d.root",
           int nExamples  = 10,      // number of examples per case
           int minHits    = 4,      // min. number of hits for a cluster to be drawn
           long maxEvents = 10000,     // -1 = all
           bool savePng   = true,
           double E_CUT   = 0.1) {
  gROOT->SetBatch(true);
  initColors();

  podio::ROOTReader reader;
  std::string in_dir = "/run/media/epic/Data/PI0/DVMP_pi0_9x130_26.07";
  std::string prefix = "DVMP_pi0_9x130";
  const auto in_file = collectInputFiles(in_dir, prefix);
  if (in_file.empty()) {
    std::cerr << "Brak plikow: " << prefix << "*.root w " << in_dir << "\n";
    return;
  }
  reader.openFiles(in_file);
  TFile* fout = new TFile(out_file.c_str(), "RECREATE");

  // ---- orphan histograms
  TH1D* h_orphDR     = new TH1D("h_orphDR",     "orphan ScFi-Img: dR;dR;par", 100, 0, 1.5);
  TH1D* h_mergedDR   = new TH1D("h_mergedDR",   "scalone ScFi-Img w Barrel: dR;dR;par", 100, 0, 1.5);
  TH1D* h_orphSameMC = new TH1D("h_orphSameMC", "orphan par: 0 rozne MC, 1 to samo MC, 2 brak asocjacji;;par", 3, -0.5, 2.5);
  TH1D* h_orphEscFi  = new TH1D("h_orphEscFi",  "orphan ScFi: E;E [GeV];N", 50, 0, 10);
  TH1D* h_orphEimg   = new TH1D("h_orphEimg",   "orphan Img: E;E [GeV];N", 50, 0, 10);
  long nPairSameMC = 0, nPairDiffMC = 0, nPairNoAssoc = 0;

  // ---- statistics for items 1 / 2
  std::map<std::pair<int, int>, long> subDist; // (nScFi, nImaging) -> number of Barrel clusters
  long nBarrelClu = 0, nSubOther = 0, nBarrelCluNoHits = 0;
  std::map<size_t, long> recCluDist, recCluDistPhoton;
  int nDebugRec = 0;
  std::map<std::string, long> presence;   // collection presence pattern, e.g. "BSI", "-SI", "--I"

  // ---- example counters
  std::map<std::string, int> nSaved;
  const std::vector<std::string> cases = {"BARREL_both", "BARREL_onlyImaging", "BARREL_onlyScFi",
                                          "ORPHAN_ScFi_and_Imaging", "NOBARREL_onlyImaging", "NOBARREL_onlyScFi",
                                          "EndcapP", "EndcapN"};
  for (auto& c : cases) nSaved[c] = 0;

  long nEv = 0, nSel = 0;
  while (auto entry = reader.readNextEntry(podio::Category::Event)) {
    ++nEv;
    if (maxEvents > 0 && nEv > maxEvents) break;
    auto frame = podio::Frame(std::move(entry));

    // ================= selection as in anal_res.cxx =================
    const auto& mc = frame.get<edm4hep::MCParticleCollection>("MCParticles");
    bool havePi0 = false;
    for (const auto& p : mc) if (p.getPDG() == 111 && p.getGeneratorStatus() == 2) { havePi0 = true; break; }
    if (!havePi0) continue;

    std::vector<McLine> mcLines;
    for (const auto& p : mc) {
      if (p.getPDG() != 22 || p.getGeneratorStatus() != 1) continue;
      bool fromPi0 = false;
      for (const auto& par : p.getParents()) if (par.getPDG() == 111) fromPi0 = true;
      if (!fromPi0) continue;
      const TVector3 mom(p.getMomentum().x, p.getMomentum().y, p.getMomentum().z);
      if (mom.Mag() < E_CUT) continue;
      if (1.3 < std::fabs(mom.Eta()) && std::fabs(mom.Eta()) < 1.6) continue;
      McLine m{TVector3(p.getVertex().x, p.getVertex().y, p.getVertex().z),
               TVector3(p.getEndpoint().x, p.getEndpoint().y, p.getEndpoint().z), mom};
      if (nSel < 20)   // endpoint check on the first events
        std::cout << "[MC gamma ev" << nEv << "] E=" << mom.Mag() << " vtx=(" << m.v.X() << "," << m.v.Y() << "," << m.v.Z()
                  << ") end=(" << m.e.X() << "," << m.e.Y() << "," << m.e.Z()
                  << ") |end-vtx|=" << (m.e - m.v).Mag() << " mm\n";
      mcLines.push_back(m);
    }
    if (mcLines.size() != 2) continue;
    ++nSel;

    const auto& bClusters     = frame.get<edm4eic::ClusterCollection>("EcalBarrelClusters");
    const auto& bClustersScFi = frame.get<edm4eic::ClusterCollection>("EcalBarrelScFiClusters");
    const auto& bClustersImg  = frame.get<edm4eic::ClusterCollection>("EcalBarrelImagingClusters");
    const auto& epClusters    = frame.get<edm4eic::ClusterCollection>("EcalEndcapPClusters");
    const auto& enClusters    = frame.get<edm4eic::ClusterCollection>("EcalEndcapNClusters");
    const auto& recs          = frame.get<edm4eic::ReconstructedParticleCollection>("ReconstructedParticles");
    const auto& recoassoc     = frame.get<edm4eic::MCRecoParticleAssociationCollection>("ReconstructedParticleAssociations");

    std::set<ID> scfiIDs, imgIDs, bIDs, epIDs, enIDs;
    for (const auto& c : bClustersScFi) scfiIDs.insert(idOf(c.getObjectID()));
    for (const auto& c : bClustersImg)  imgIDs.insert(idOf(c.getObjectID()));
    for (const auto& c : bClusters)     bIDs.insert(idOf(c.getObjectID()));
    for (const auto& c : epClusters)    epIDs.insert(idOf(c.getObjectID()));
    for (const auto& c : enClusters)    enIDs.insert(idOf(c.getObjectID()));

    // ---- cluster status: used by reco / matched track / improved
    std::set<ID> usedByReco, trackMatched;
    for (const auto& rp : recs) for (const auto& c : rp.getClusters()) markCluster(c, usedByReco);
    for (const char* name : {"EcalBarrelTrackClusterMatches", "EcalEndcapPTrackClusterMatches", "EcalEndcapNTrackClusterMatches"})
      for (const auto& m : frame.get<edm4eic::TrackClusterMatchCollection>(name))
        if (m.getCluster().isAvailable()) markCluster(m.getCluster(), trackMatched);
    auto status = [&](const edm4eic::Cluster& c) -> std::string {
      const ID k = idOf(c.getObjectID());
      if (usedByReco.count(k))   return "reco";
      if (trackMatched.count(k)) return "track";
      return "improved";
    };

    // ---- cluster -> MC associations (best weight)
    std::map<ID, std::pair<double, ID>> best;
    for (const char* an : {"EcalBarrelScFiClusterAssociations", "EcalBarrelImagingClusterAssociations",
                           "EcalBarrelClusterAssociations"})
      for (const auto& a : frame.get<edm4eic::MCRecoClusterParticleAssociationCollection>(an)) {
        if (!a.getRec().isAvailable() || !a.getSim().isAvailable()) continue;
        const ID rk = idOf(a.getRec().getObjectID());
        auto it = best.find(rk);
        if (it == best.end() || a.getWeight() > it->second.first)
          best[rk] = {a.getWeight(), idOf(a.getSim().getObjectID())};
      }

    auto mkExtras = [&](std::vector<std::pair<TVector3, std::string>> stars) {
      Extras ex; ex.mc = mcLines; ex.stars = std::move(stars); return ex;
    };

    // ---- collection presence statistics
    presence[std::string(bClusters.empty() ? "-" : "B") + (bClustersScFi.empty() ? "-" : "S") +
             (bClustersImg.empty() ? "-" : "I")]++;

    // ---- cases without Barrel: only one technology
    auto drawSingle = [&](const edm4eic::ClusterCollection& cc, const std::string& cs, int family, const char* lab) {
      if (nSaved[cs] >= nExamples) return;
      Group g{lab, family, {}};
      std::set<ID> seen;
      std::vector<std::pair<TVector3, std::string>> stars;
      for (const auto& c : cc) { gatherHits(c, g.hits, seen); stars.push_back({posOf(c), std::string(lab) + " " + status(c)}); }
      if (g.hits.size() < (size_t)minHits) return;
      const std::string name  = Form("c3d_%s_%d", cs.c_str(), nSaved[cs]);
      const std::string title = Form("%s ev%ld: %zu klastrow, bez Barrel i bez drugiej technologii", cs.c_str(), nEv, cc.size());
      if (drawExample(fout, name, title, {g}, savePng, mkExtras(stars))) ++nSaved[cs];
    };
    if (bClusters.empty() && bClustersScFi.empty() && !bClustersImg.empty())
      drawSingle(bClustersImg, "NOBARREL_onlyImaging", 1, "Imaging");
    if (bClusters.empty() && bClustersImg.empty() && !bClustersScFi.empty())
      drawSingle(bClustersScFi, "NOBARREL_onlyScFi", 0, "ScFi");

    // ======================= ITEM 1: constituents of EcalBarrelClusters =======================
    std::set<ID> usedScFi, usedImg;
    for (const auto& c : bClusters) {
      ++nBarrelClu;
      int nS = 0, nI = 0;
      Group gS{"ScFi", 0, {}}, gI{"Imaging", 1, {}};
      std::set<ID> seenS, seenI;
      std::vector<edm4eic::Cluster> subs;
      for (const auto& sub : c.getClusters()) subs.push_back(sub);
      const edm4eic::Cluster* scPtr = nullptr; const edm4eic::Cluster* imPtr = nullptr;
      for (const auto& sub : subs) {
        const ID key = idOf(sub.getObjectID());
        if (scfiIDs.count(key))     { ++nS; usedScFi.insert(key); gatherHits(sub, gS.hits, seenS); scPtr = &sub; }
        else if (imgIDs.count(key)) { ++nI; usedImg.insert(key);  gatherHits(sub, gI.hits, seenI); imPtr = &sub; }
        else ++nSubOther;
      }
      subDist[{nS, nI}]++;
      if (c.getHits().empty() && gS.hits.empty() && gI.hits.empty()) ++nBarrelCluNoHits;
      if (scPtr && imPtr) h_mergedDR->Fill(dRpos(posOf(*scPtr), posOf(*imPtr)));

      const std::string cs = (nS > 0 && nI > 0) ? "BARREL_both" : (nI > 0) ? "BARREL_onlyImaging" : (nS > 0) ? "BARREL_onlyScFi" : "";
      if (!cs.empty() && nSaved[cs] < nExamples && gS.hits.size() + gI.hits.size() >= (size_t)minHits) {
        std::vector<Group> gs;
        if (!gS.hits.empty()) gs.push_back(gS);
        if (!gI.hits.empty()) gs.push_back(gI);
        const std::string name  = Form("c3d_%s_%d", cs.c_str(), nSaved[cs]);
        const std::string title = Form("%s ev%ld [%s]: E_{clu}=%.2f GeV (nScFi=%d, nImg=%d)",
                                       cs.c_str(), nEv, status(c).c_str(), c.getEnergy(), nS, nI);
        if (drawExample(fout, name, title, gs, savePng, mkExtras({{posOf(c), "Barrel"}}))) ++nSaved[cs];
      }
    }

    // ======================= orphans: ScFi / Imaging outside a Barrel cluster =======================
    {
      std::vector<edm4eic::Cluster> orphS, orphI;
      for (const auto& c : bClustersScFi) if (!usedScFi.count(idOf(c.getObjectID()))) orphS.push_back(c);
      for (const auto& c : bClustersImg)  if (!usedImg.count(idOf(c.getObjectID())))  orphI.push_back(c);
      for (const auto& s : orphS) h_orphEscFi->Fill(s.getEnergy());
      for (const auto& im : orphI) h_orphEimg->Fill(im.getEnergy());

      for (const auto& s : orphS)
        for (const auto& im : orphI) {
          const double d = dRpos(posOf(s), posOf(im));
          h_orphDR->Fill(d);
          auto is = best.find(idOf(s.getObjectID())), ii = best.find(idOf(im.getObjectID()));
          const bool both = (is != best.end() && ii != best.end());
          if (!both) { h_orphSameMC->Fill(2); ++nPairNoAssoc; }
          else if (is->second.second == ii->second.second) { h_orphSameMC->Fill(1); ++nPairSameMC; }
          else { h_orphSameMC->Fill(0); ++nPairDiffMC; }
          if (d < 0.3) {
            std::cout << "[ORPHAN ev" << nEv << "] dR=" << d << " E_scfi=" << s.getEnergy() << " E_img=" << im.getEnergy()
                      << " nHit(scfi/img)=" << s.getHits().size() << "/" << im.getHits().size()
                      << " status(scfi/img)=" << status(s) << "/" << status(im)
                      << " MCmatch=" << (both ? (is->second.second == ii->second.second ? "ta sama" : "rozne") : "brak");
            if (is != best.end()) std::cout << " w_scfi=" << is->second.first;
            if (ii != best.end()) std::cout << " w_img=" << ii->second.first;
            std::cout << "\n";
          }
        }

      const std::string cs = "ORPHAN_ScFi_and_Imaging";
      if (!orphS.empty() && !orphI.empty() && nSaved[cs] < nExamples) {
        Group gS{"ScFi orphan", 0, {}}, gI{"Imaging orphan", 1, {}};
        std::set<ID> seenS, seenI;
        std::vector<std::pair<TVector3, std::string>> stars;
        for (const auto& c : orphS) { gatherHits(c, gS.hits, seenS); stars.push_back({posOf(c), "ScFi " + status(c)}); }
        for (const auto& c : orphI) { gatherHits(c, gI.hits, seenI); stars.push_back({posOf(c), "Img " + status(c)}); }
        if (gS.hits.size() + gI.hits.size() >= (size_t)minHits) {
          const std::string name  = Form("c3d_%s_%d", cs.c_str(), nSaved[cs]);
          const std::string title = Form("%s ev%ld (orphan ScFi=%zu, orphan Img=%zu, BarrelClusters=%zu)",
                                         cs.c_str(), nEv, orphS.size(), orphI.size(), bClusters.size());
          if (drawExample(fout, name, title, {gS, gI}, savePng, mkExtras(stars))) ++nSaved[cs];
        }
      }
    }

    // ======================= Endcaps =======================
    auto doEndcap = [&](const edm4eic::ClusterCollection& cc, const std::string& cs) {
      for (const auto& c : cc) {
        if (nSaved[cs] >= nExamples) break;
        Group g{cs, 2, {}};
        std::set<ID> seen;
        gatherHits(c, g.hits, seen);
        if (g.hits.size() < (size_t)minHits) continue;
        const std::string name  = Form("c3d_%s_%d", cs.c_str(), nSaved[cs]);
        const std::string title = Form("%s ev%ld [%s]: E_{clu}=%.2f GeV", cs.c_str(), nEv, status(c).c_str(), c.getEnergy());
        if (drawExample(fout, name, title, {g}, savePng, mkExtras({{posOf(c), cs}}))) ++nSaved[cs];
      }
    };
    doEndcap(epClusters, "EndcapP");
    doEndcap(enClusters, "EndcapN");

    // ======================= ITEM 2: ReconstructedParticles -> getClusters() =======================
    auto origin = [&](const edm4eic::Cluster& c) -> std::string {
      const ID k = idOf(c.getObjectID());
      if (bIDs.count(k))    return "EcalBarrelClusters";
      if (epIDs.count(k))   return "EcalEndcapPClusters";
      if (enIDs.count(k))   return "EcalEndcapNClusters";
      if (scfiIDs.count(k)) return "BarrelScFi(!)";
      if (imgIDs.count(k))  return "BarrelImaging(!)";
      return "inna";
    };
    for (const auto& r : recs) {
      const size_t n = r.getClusters().size();
      recCluDist[n]++;
      if (n >= 2 && nDebugRec < 15) {
        ++nDebugRec;
        std::cout << "[DEBUG ev" << nEv << "] ReconstructedParticle E=" << r.getEnergy() << " ma " << n << " klastrow:";
        for (const auto& c : r.getClusters())
          std::cout << " [" << origin(c) << " E=" << c.getEnergy() << " sub=" << c.getClusters().size() << "]";
        std::cout << "\n";
      }
    }
    for (const auto& a : recoassoc) {
      if (a.getSim().getPDG() != 22) continue;
      recCluDistPhoton[a.getRec().getClusters().size()]++;
    }
  }

  fout->cd();
  fout->Write();
  fout->Close();

  // ======================= printout =======================
  std::cout << "\nWybrane zdarzenia (pi0 -> 2 fotony MC): " << nSel << " / " << nEv << "\n";
  std::cout << "\n=== PKT 1: podklastry w EcalBarrelClusters (" << nBarrelClu << " klastrow) ===\n";
  std::cout << "  (nScFi, nImaging) -> liczba klastrow\n";
  for (auto& [k, n] : subDist)
    std::cout << "   (" << k.first << ", " << k.second << ")  " << std::setw(8) << n
              << "  (" << std::fixed << std::setprecision(2) << (nBarrelClu ? 100.0 * n / nBarrelClu : 0) << "%)\n";
  std::cout << "  Podklastry, ktore nie sa ani ScFi ani Imaging: " << nSubOther << "\n";
  std::cout << "  Klastry Barrel bez zadnych hitow: " << nBarrelCluNoHits << "\n";

  auto printDist = [](const char* title, const std::map<size_t, long>& m) {
    long tot = 0; for (auto& [k, n] : m) tot += n;
    std::cout << "\n=== " << title << " (razem " << tot << ") ===\n";
    for (auto& [k, n] : m)
      std::cout << "   getClusters().size() = " << k << " : " << std::setw(8) << n
                << "  (" << std::fixed << std::setprecision(2) << (tot ? 100.0 * n / tot : 0) << "%)\n";
  };
  printDist("PKT 2: ReconstructedParticles - wszystkie", recCluDist);
  printDist("PKT 2: czastki zasocjowane z MC fotonem (PDG 22)", recCluDistPhoton);

  std::cout << "\n=== PKT 3: zapisane rysunki 3D ===\n";
  for (auto& c : cases) std::cout << "   " << std::setw(26) << std::left << c << std::right << nSaved[c] << " / " << nExamples << "\n";

  std::cout << "\n=== PKT 4: pary orphanow ScFi-Imaging ===\n";
  std::cout << "  to samo MC " << nPairSameMC << ", rozne MC " << nPairDiffMC << ", brak asocjacji " << nPairNoAssoc << "\n";

  std::cout << "\n=== PKT 5: obecnosc kolekcji w evencie (B=Barrel, S=ScFi, I=Imaging) ===\n";
  for (auto& [k, n] : presence)
    std::cout << "   " << k << "  " << std::setw(8) << n << "  (" << std::fixed << std::setprecision(2)
              << (nSel ? 100.0 * n / nSel : 0) << "%)\n";

  std::cout << "Zapisano: " << out_file << "  (TCanvas c3d_*; otworz: root -l " << out_file << " -> new TBrowser)\n";
}
