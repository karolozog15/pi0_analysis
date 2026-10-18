#include <TFile.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TLine.h>
#include <TStyle.h>
#include <TMath.h>
#include <TLatex.h>
#include <TPad.h>
#include <TGaxis.h>
#include <TSystem.h>
#include <iostream>
#include <string>
#include <algorithm>

static const char* kDir = "./plots_simple";

// =============================================================================
// STYLE
// =============================================================================
static void SetGlobalStyle() {
  gStyle->SetLineWidth(3);
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  gStyle->SetImageScaling(2.0);
  TH1::SetDefaultSumw2(kTRUE);
  TGaxis::SetMaxDigits(3);
}

static void StyleHist(TH1* h, Color_t color, int lw = 3, int ls = 1) {
  if (!h) return;
  h->SetLineColor(color); h->SetMarkerColor(color);
  h->SetLineWidth(lw);    h->SetLineStyle(ls);
}

static void StyleFrame(TH1* h, const char* xTitle, const char* yTitle) {
  if (!h) return;
  h->GetXaxis()->SetTitle(xTitle);  h->GetYaxis()->SetTitle(yTitle);
  h->GetXaxis()->SetLabelFont(42);  h->GetYaxis()->SetLabelFont(42);
  h->GetXaxis()->SetTitleFont(42);  h->GetYaxis()->SetTitleFont(42);
  h->GetXaxis()->SetLabelSize(0.050); h->GetYaxis()->SetLabelSize(0.050);
  h->GetXaxis()->SetTitleSize(0.058); h->GetYaxis()->SetTitleSize(0.058);
  h->GetXaxis()->SetTitleOffset(0.95); h->GetYaxis()->SetTitleOffset(1.35);
  h->GetXaxis()->SetNdivisions(505);
}

static TLegend* MakeLegend(double x1, double y1, double x2, double y2, double ts = 0.045) {
  TLegend* leg = new TLegend(x1, y1, x2, y2);
  leg->SetBorderSize(0); leg->SetFillStyle(0);
  leg->SetTextFont(42);  leg->SetTextSize(ts);
  return leg;
}

static void SetPad1D(TVirtualPad* pad, bool logy = false) {
  pad->SetLeftMargin(0.15);  pad->SetBottomMargin(0.14);
  pad->SetRightMargin(0.035); pad->SetTopMargin(0.07);
  pad->SetLogy(logy ? 1 : 0);
}

// etykieta ePIC na biezacym padzie
static void DrawEPICLabel(double x = 0.20, double y = 0.86, double ts = 0.045) {
  TLatex* l = new TLatex();
  l->SetNDC(); l->SetTextFont(42); l->SetTextSize(ts);
  l->DrawLatex(x, y,        "#bf{ePIC simulation}");
  l->DrawLatex(x, y - 0.05, "26.07.01 Campaign");
  l->DrawLatex(x, y - 0.10, "e+p DV#pi^{0}P 9x130 GeV");
}

// nowy canvas -> rysowanie -> zapis PNG -> usuniecie
template <typename F>
static void save(const std::string& name, int w, int h, F draw) {
  TCanvas* c = new TCanvas(("c_" + name).c_str(), name.c_str(), w, h);
  draw();
  c->SaveAs(Form("%s/%s.png", kDir, name.c_str()));
  delete c;
}

template <typename T>
static T* GetHist(TFile* f, const char* name) {
  T* h = (T*)f->Get(name);
  if (!h) std::cerr << "WARNING: histogram '" << name << "' not found!\n";
  else    h->SetDirectory(0);
  return h;
}

// =============================================================================
static const int kNG = 5;
static const char* kStageLabels[kNG] = {"1: P+P", "1+2: + split P", "1-3: + P+I, I+I",
                                        "1-4: + split I", "1-5: + P+H, I+H, H+H"};
// kolory (linia i wypelnienie) kolejnych etapow - kazdy inny
// kolory etapow: P+P niebieski, split P czerwony, P+I/I+I pomaranczowy, split I fioletowy, hity zielony
static const int kStageColors[kNG] = {kBlue + 1, kRed + 1, kOrange + 1, kViolet + 1, kGreen + 2};
static const double kStageAlpha[kNG] = {1.0, 1.0, 1.0, 1.0, 1.0};
static const char* kCat[3]      = {"split", "hits", "improved"};
static const char* kCatLabel[3] = {"Split", "#geq 1 #gamma from hits", "#geq 1 #gamma improved"};

// ---------------------------------------------------------------- 1) liczba klastrow (os X: 0-10)
static void drawCount(TFile* f, const char* hname, const char* png, int color) {
  TH1D* h = GetHist<TH1D>(f, hname);
  if (!h) return;
  save(png, 900, 800, [&]() {
    SetPad1D(gPad, true);
    StyleHist(h, color);
    h->SetFillColorAlpha(color, 0.30);
    StyleFrame(h, "N clusters per event", "Events");
    h->GetXaxis()->SetRangeUser(-0.5, 10.5);
    h->SetMinimum(0.5);
    h->SetMaximum(h->GetMaximum() * 8.0);
    h->Draw("HIST");
    gPad->RedrawAxis();
    DrawEPICLabel();
  });
  std::cout << Form("%-12s srednia = %.2f\n", hname, h->GetMean());
}

// ---------------------------------------------------------------- 2) skumulowane
static void drawCum(TFile* f, const char* base, const char* png, const char* xTitle) {
  TH1D* h[kNG];
  double ymax = 0;
  for (int s = 0; s < kNG; ++s) {
    h[s] = GetHist<TH1D>(f, Form("%s_%d", base, s));
    if (!h[s]) return;
    ymax = std::max(ymax, h[s]->GetMaximum());
  }
  save(png, 900, 800, [&]() {
    SetPad1D(gPad, false);
    for (int s = 0; s < kNG; ++s) {
      StyleHist(h[s], kStageColors[s], 3);
      if (kStageAlpha[s] > 0) { h[s]->SetFillColorAlpha(kStageColors[s], kStageAlpha[s]); h[s]->SetFillStyle(1001); }
    }
    StyleFrame(h[kNG - 1], xTitle, "Events");
    h[kNG - 1]->SetMaximum(1.25 * ymax);
    h[kNG - 1]->SetMinimum(0);
    h[kNG - 1]->Draw("HIST");                                   // najwiekszy najpierw
    for (int s = kNG - 2; s >= 0; --s) h[s]->Draw("HIST SAME"); // mniejsze na wierzchu
    TLegend* leg = MakeLegend(0.2, 0.55, 0.4, 0.90, 0.040);
    for (int s = kNG - 1; s >= 0; --s)
      leg->AddEntry(h[s], Form("%s", kStageLabels[s]), "f");
    leg->Draw();
    gPad->RedrawAxis();
    DrawEPICLabel(0.2,0.4);
  });
}

// ---------------------------------------------------------------- 3) wydajnosc: MC vs accepted + ratio pod spodem
static void drawEfficiency(TFile* f,std::string val) {
  TH1D* hMC  = GetHist<TH1D>(f, Form("mcPi0_%s", val.c_str()));
  TH1D* hAcc = GetHist<TH1D>(f, Form("accPi0_%s", val.c_str()));
  if (!hMC || !hAcc) return;

  const double etaLo = 1.3, etaHi = 1.6;   // pomijany zakres (szczelina)

  // wydajnosc calkowita bez binow, ktorych srodek lezy w [etaLo, etaHi]
  double nMC = 0, nAcc = 0;
  nMC=hMC->GetEntries();
  nAcc=hAcc->GetEntries();
  const double eff = nMC > 0 ? 100.0 * nAcc / nMC : 0;

  // dwie pionowe linie na biezacym padzie (od dolu do gory ramki)
  auto drawGapLines = [&](TH1* h) {
    const double y1 = h->GetMinimum(), y2 = h->GetMaximum();
    for (double x : {etaLo, etaHi}) {
      TLine* l = new TLine(x, y1, x, y2);
      l->SetLineColor(kGray + 2); l->SetLineStyle(2); l->SetLineWidth(3);
      l->Draw();
    }
  };

  save(Form("eta_mc_vs_acc_%s", val.c_str()), 900, 900, [&]() {
    TVirtualPad* cv = gPad;
    TPad* pTop = new TPad("pTop", "", 0, 0.30, 1, 1.0);
    TPad* pBot = new TPad("pBot", "", 0, 0.00, 1, 0.30);
    pTop->SetLeftMargin(0.15);  pTop->SetRightMargin(0.035);
    pTop->SetTopMargin(0.07);   pTop->SetBottomMargin(0.02);
    pBot->SetLeftMargin(0.15);  pBot->SetRightMargin(0.035);
    pBot->SetTopMargin(0.03);   pBot->SetBottomMargin(0.38);
    pTop->Draw(); pBot->Draw();

    // ---- gora: MC i accepted ----
    pTop->cd();
    pTop->SetLogy();
    StyleHist(hMC,  kRed + 1);
    StyleHist(hAcc, kBlue + 1);
    StyleFrame(hMC, "", "Events");
    hMC->GetXaxis()->SetLabelSize(0);
    hMC->GetYaxis()->SetTitleOffset(1.35);
    hMC->SetMinimum(0.5);
    hMC->SetMaximum(2.0* std::max(hMC->GetMaximum(), hAcc->GetMaximum()));
    hMC->Draw("HIST");
    hAcc->Draw("HIST SAME");
    //drawGapLines(hMC);
    TLegend* leg = MakeLegend(0.60, 0.55, 0.75, 0.72);
    leg->AddEntry(hMC,  "MC truth", "l");
    leg->AddEntry(hAcc, "MC accepted", "l");
    leg->Draw();
    gPad->RedrawAxis();
    
    DrawEPICLabel(0.6,0.85);
    TLatex* t = new TLatex();
    t->SetNDC();
    t->SetTextFont(42);
    t->SetTextSize(0.045);
    t->DrawLatex(0.60, 0.40, Form("Efficiency: %.1f%%", eff));
    t->DrawLatex(0.60, 0.35, Form("E_{#gamma} > 1 GeV"));
    t->DrawLatex(0.60, 0.30, Form("#eta_{#gamma} < 1.3 or > 1.6", nAcc));

    // ---- dol: ratio ----
    pBot->cd();
    TH1D* r = (TH1D*)hAcc->Clone("ratio_eta");
    r->SetDirectory(0);
    r->Divide(hAcc, hMC, 1, 1, "B");
    StyleHist(r, kBlack, 2);
    r->SetMarkerStyle(20); r->SetMarkerSize(0.8);
    StyleFrame(r, "#eta_{#pi^{0}}", "Acc / MC");
    r->GetXaxis()->SetLabelSize(0.115); r->GetXaxis()->SetTitleSize(0.135);
    r->GetYaxis()->SetLabelSize(0.100); r->GetYaxis()->SetTitleSize(0.115);
    r->GetXaxis()->SetTitleOffset(1.0); r->GetYaxis()->SetTitleOffset(0.55);
    r->GetYaxis()->SetNdivisions(505);
    r->SetMinimum(0); r->SetMaximum(1.2);
    r->Draw("E");
    TLine* one = new TLine(r->GetXaxis()->GetXmin(), 1.0, r->GetXaxis()->GetXmax(), 1.0);
    one->SetLineStyle(2); one->SetLineWidth(2); one->Draw();
    //drawGapLines(r);
    gPad->RedrawAxis();

    cv->cd();
  });

  std::cout << Form("Wydajnosc calkowita (bez eta w [%.2f, %.2f]): %.0f / %.0f = %.1f%%\n",
                    etaLo, etaHi, nAcc, nMC, eff);
}

// ---------------------------------------------------------------- 4) 2D MC vs reko
static void draw2D(TFile* f, const char* var, const char* cat, const char* label, const char* xT, const char* yT) {
  TH2D* h = GetHist<TH2D>(f, Form("h2_pi0_%s_%s", var, cat));
  if (!h) return;
  save(Form("2d_%s_%s", var, cat), 900, 800, [&]() {
    gPad->SetLeftMargin(0.14);  gPad->SetBottomMargin(0.14);
    gPad->SetRightMargin(0.17); gPad->SetTopMargin(0.05);
    gPad->SetLogz(1);
    StyleFrame(h, xT, yT);
    h->GetYaxis()->SetTitleOffset(1.2);
    h->GetZaxis()->SetTitle("Events");
    h->GetZaxis()->SetLabelSize(0.045); h->GetZaxis()->SetTitleSize(0.050);
    h->GetZaxis()->SetTitleOffset(1.1); h->GetZaxis()->SetNdivisions(808);
    h->Draw("COLZ");
    const double lo = std::max(h->GetXaxis()->GetXmin(), h->GetYaxis()->GetXmin());
    const double hi = std::min(h->GetXaxis()->GetXmax(), h->GetYaxis()->GetXmax());
    TLine* ln = new TLine(lo, lo, hi, hi);
    ln->SetLineColor(kRed + 1); ln->SetLineStyle(2); ln->SetLineWidth(3);
    ln->Draw();
    gPad->RedrawAxis();
    TLatex t; t.SetNDC(); t.SetTextFont(42); t.SetTextSize(0.050);
    t.DrawLatex(0.17, 0.88, label);
    t.SetTextSize(0.042);
    //t.DrawLatex(0.17, 0.82, Form("N = %.0f,  #rho = %.2f", h->GetEntries(), h->GetCorrelationFactor()));
  });
}

// ---------------------------------------------------------------- 5) good / w oknie per grupa: counts na gorze, fraction na dole
static void drawGood(TFile* f) {
  TH1D* hWin  = GetHist<TH1D>(f, "h_grp");
  TH1D* hGood = GetHist<TH1D>(f, "h_grpGood");
  if (!hWin || !hGood) return;

  save("good",1000, 1000, [&]() {
    TVirtualPad* cv = gPad;
    TPad* pTop = new TPad("pTop", "", 0, 0.30, 1, 1.0);
    TPad* pBot = new TPad("pBot", "", 0, 0.00, 1, 0.30);
    pTop->SetLeftMargin(0.15);  pTop->SetRightMargin(0.1);
    pTop->SetTopMargin(0.07);   pTop->SetBottomMargin(0.02);
    pBot->SetLeftMargin(0.15);  pBot->SetRightMargin(0.1);
    pBot->SetTopMargin(0.03);   pBot->SetBottomMargin(0.25);
    pTop->Draw(); pBot->Draw();

    // ---- gora: liczby ----
    pTop->cd();
    StyleHist(hWin,  kBlue + 1);   hWin->SetFillColorAlpha(kBlue + 1, 0.30);
    StyleHist(hGood, kGreen + 2);  hGood->SetFillColorAlpha(kGreen + 2, 0.50);
    StyleFrame(hWin, "", "Number of #pi^{0}");
    hWin->GetXaxis()->SetLabelSize(0);   // etykiety grup tylko na dolnym padzie
    hWin->SetMinimum(0);
    hWin->SetMaximum(1.3 * std::max(hWin->GetMaximum(), hGood->GetMaximum()));
    hWin->Draw("HIST");
    hGood->Draw("HIST SAME");
    TLegend* leg = MakeLegend(0.40, 0.62, 0.85, 0.90, 0.045);
  leg->AddEntry(hWin,  "#splitline{#pi^{0} in mass window (3#sigma)}{obtained in each method}", "f");
leg->AddEntry(hGood, "#splitline{Good direction}{#DeltaR(#pi^{0}_{reco}, #pi^{0}_{MC}) < 0.2}", "f");
    leg->Draw();
    gPad->RedrawAxis();

    // ---- dol: odsetek good ----
    pBot->cd();
    TH1D* hFrac = (TH1D*)hGood->Clone("h_grpFrac");
    hFrac->SetDirectory(0);
    hFrac->Divide(hGood, hWin, 1, 1, "B");
    hFrac->Scale(100.0);
    StyleHist(hFrac, kOrange + 7);
    hFrac->SetFillColorAlpha(kOrange + 1, 0.40);
    hFrac->SetMarkerStyle(20); hFrac->SetMarkerSize(1.0);
    StyleFrame(hFrac, "", "Good / window [%]");
    hFrac->GetXaxis()->SetLabelSize(0.15);
    hFrac->GetYaxis()->SetLabelSize(0.075); hFrac->GetYaxis()->SetTitleSize(0.12);
    hFrac->GetYaxis()->SetTitleOffset(0.6);
    hFrac->GetYaxis()->SetNdivisions(505);
    hFrac->SetMinimum(0); hFrac->SetMaximum(135);
    hFrac->Draw("HIST");
    hFrac->Draw("E1 SAME");
    TLatex txt; txt.SetTextAlign(21); txt.SetTextFont(42); txt.SetTextSize(0.1);
    std::cout << "\nGrupa            w oknie    good   good/w oknie\n";
    for (int i = 1; i <= hFrac->GetNbinsX(); ++i) {
      const double nWin = hWin->GetBinContent(i), nGood = hGood->GetBinContent(i);
      if (nWin <= 0) continue;
      const double pct = hFrac->GetBinContent(i);
      txt.DrawLatex(hFrac->GetBinCenter(i), pct + 4 + hFrac->GetBinError(i), Form("%.1f%%", pct));
      std::cout << Form("%-14s %8.0f %8.0f   %6.1f%%\n", hWin->GetXaxis()->GetBinLabel(i), nWin, nGood, pct);
    }
    gPad->RedrawAxis();
    cv->cd();
  });
}
// =============================================================================
void draw(const std::string& in_file = "pi0_simple_old.root") {
  SetGlobalStyle();
  gSystem->mkdir(kDir, kTRUE);
  TFile* f = TFile::Open(in_file.c_str());
  if (!f || f->IsZombie()) { std::cerr << "Nie mozna otworzyc " << in_file << "\n"; return; }

  drawCount(f, "h_nUnused",   "n_unused",   kAzure + 1);
  drawCount(f, "h_nImproved", "n_improved", kOrange + 7);
  drawCount(f, "h_nNewCl",    "n_newcl",    kMagenta + 1);

  drawCum(f, "cum_pi0_eta", "cum_eta",  "#eta_{#pi^{0}}");
  drawCum(f, "cum_pi0_m",   "cum_mass", "m_{#gamma#gamma} [GeV]");

  drawEfficiency(f, "eta");
  drawEfficiency(f, "E");

  for (int i = 0; i < 3; ++i) {
    draw2D(f, "E",   kCat[i], kCatLabel[i], "E_{#pi^{0}}^{MC} [GeV]", "E_{#pi^{0}}^{reco} [GeV]");
    draw2D(f, "eta", kCat[i], kCatLabel[i], "#eta_{#pi^{0}}^{MC}",    "#eta_{#pi^{0}}^{reco}");
  }

  drawGood(f);
  std::cout << "\nWszystkie wykresy zapisane w " << kDir << "/\n";
}
