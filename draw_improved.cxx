// =============================================================================
// draw_improved.cxx - wszystkie wykresy z pi0_improved.root (wyjscie anal.cxx)
//   ./plots_improved : wykresy 1D (porownania MC/final, rozdzielczosci, metody, skumulowane, klastry)
//   ./plots_2d       : histogramy 2D MC vs reko, bias/rozdzielczosc w funkcji MC
// Metody: 0 PP, 1 SplitP, 2 PI, 3 II, 4 SplitI, 5 PH, 6 IH, 7 HH
// Kazde final pi0 pochodzi z dokladnie jednej metody (kaskada grup, reset kandydata
// przed kazda grupa), wiec histogramy metod sa rozlaczne, a etapy skumulowane to
// suma metod do ostatniej metody danej grupy:
//   1 | 1+2 | 1+2+3 | 1+2+3+4 | 1+2+3+4+5
// Uzycie:  root -l -q 'draw_improved.cxx("pi0_improved.root")'
// =============================================================================
#include <TFile.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TLatex.h>
#include <TLine.h>
#include <TF1.h>
#include <TGraphErrors.h>
#include <TMultiGraph.h>
#include <TStyle.h>
#include <TSystem.h>
#include <TString.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdio>

static const int kNM = 8;
static const char* kTags[kNM]   = {"PP", "SplitP", "PI", "II", "SplitI", "PH", "IH", "HH"};
static const char* kLabels[kNM] = {"P+P", "Split P", "P+I", "I+I", "Split I", "P+H", "I+H", "H+H"};

// etapy (grupy): ostatnia metoda kazdej grupy
static const int kNG = 5;
static const int kGroupLast[kNG] = {0, 1, 3, 4, 7};
static const char* kStageLabels[kNG] = {"1: P+P", "1+2: + split P", "1+..+3: + P+I, I+I",
                                        "1+..+4: + split I", "1+..+5: + P+H, I+H, H+H"};
static const int kStageColors[kNG] = {kBlack, kRed + 1, kBlue + 1, kMagenta + 1, kGreen + 2};

// 2D
static const int kNCls = 4, kNTech = 5;
static const char* clsTag[kNCls]   = {"P", "I", "S", "H"};
static const char* techTag[kNTech] = {"barrel", "imaging", "scfi", "endcapP", "endcapN"};
static const char* grpTag[kNG]     = {"PP", "SplitP", "PI_II", "SplitI", "PH_IH_HH"};
static const char* grpLab[kNG]     = {"P+P", "split P", "P+I, I+I", "split I", "P+H, I+H, H+H"};
static const char* varTag[3]       = {"E", "eta", "phi"};

// ---------------------------------------------------------------- pomocnicze 1D
static TH1D* getMethodHist(TFile* f, int k, const std::string& part, const std::string& var) {
  return (TH1D*)f->Get(Form("final_%s_%s_%s", kTags[k], part.c_str(), var.c_str()));
}

static void overlay(TFile* f, const std::string& part, const std::string& var) {
  TH1D* hMC  = (TH1D*)f->Get(("mc"    + part + "_" + var).c_str());
  TH1D* hFin = (TH1D*)f->Get(("final" + part + "_" + var).c_str());
  if (!hMC || !hFin) { std::cout << "Brak histogramu: " << part << "_" << var << "\n"; return; }
  hMC->SetLineColor(kBlue + 1);  hMC->SetLineWidth(2);
  hFin->SetLineColor(kRed + 1);  hFin->SetLineWidth(2);
  hMC->SetMaximum(1.2 * std::max(hMC->GetMaximum(), hFin->GetMaximum()));
  hMC->SetTitle((part + " " + var).c_str());
  hMC->Draw("HIST");
  hFin->Draw("HIST SAME");
  TLegend* leg = new TLegend(0.30, 0.72, 0.59, 0.88);
  leg->SetBorderSize(0);
  leg->AddEntry(hMC,  Form("MC (%.0f)", hMC->GetEntries()), "l");
  leg->AddEntry(hFin, Form("final (%.0f)", hFin->GetEntries()), "l");
  leg->Draw();
}

static void drawRatio(TFile* f, const std::string& part, const std::string& var) {
  TH1D* hMC  = (TH1D*)f->Get(("mc"    + part + "_" + var).c_str());
  TH1D* hFin = (TH1D*)f->Get(("acc" + part + "_" + var).c_str());
  if (!hMC || !hFin) { std::cout << "Brak histogramu (ratio): " << part << "_" << var << "\n"; return; }
  TH1D* r = (TH1D*)hFin->Clone(("ratio_" + part + "_" + var).c_str());
  r->SetDirectory(0);
  r->Divide(hFin, hMC, 1, 1, "B");
  r->SetTitle((part + " " + var + ": final / MC").c_str());
  r->GetYaxis()->SetTitle("final / MC");
  r->SetMinimum(0); r->SetMaximum(1.2);
  r->SetMarkerStyle(20); r->SetLineColor(kBlack);
  r->Draw("E");
}

// kazda metoda na osobnym padzie (siatka 2x4)
static void drawEachMethod(TFile* f, const std::string& part, const std::string& var,
                           const std::string& xTitle, const std::string& outPng) {
  TCanvas* c = new TCanvas(Form("c_%s_%s", part.c_str(), var.c_str()), outPng.c_str(), 1200, 1800);
  c->Divide(2, 4);
  for (int k = 0; k < kNM; ++k) {
    c->cd(k + 1);
    TH1D* h = getMethodHist(f, k, part, var);
    if (!h) { std::cout << "Brak histogramu: final_" << kTags[k] << "_" << part << "_" << var << "\n"; continue; }
    h->SetTitle(Form("%s  (N=%.0f);%s;N", kLabels[k], h->GetEntries(), xTitle.c_str()));
    h->SetLineWidth(2);
    h->SetLineColor(kAzure + 1);
    h->SetFillColorAlpha(kAzure + 1, 0.3);
    h->Draw("HIST");
  }
  c->SaveAs(outPng.c_str());
}

// suma metod 0..kGroupLast[s] (metody sa rozlaczne, wiec to liczba pi0 znalezionych do etapu s)
static TH1D* cumulativeHist(TFile* f, const std::string& part, const std::string& var, int s) {
  TH1D* sum = nullptr;
  for (int k = 0; k <= kGroupLast[s]; ++k) {
    TH1D* h = getMethodHist(f, k, part, var);
    if (!h) continue;
    if (!sum) {
      sum = (TH1D*)h->Clone(Form("cum_%s_%s_%d", part.c_str(), var.c_str(), s));
      sum->SetDirectory(0);
    } else sum->Add(h);
  }
  return sum;
}

static void drawCumulative(TFile* f, const std::string& part, const std::string& var, const std::string& xTitle) {
  TH1D* h[kNG];
  double ymax = 0;
  for (int s = 0; s < kNG; ++s) {
    h[s] = cumulativeHist(f, part, var, s);
    if (!h[s]) { std::cout << "Brak danych: " << part << "_" << var << "\n"; return; }
    h[s]->SetLineColor(kStageColors[s]);
    h[s]->SetLineWidth(2);
    ymax = std::max(ymax, h[s]->GetMaximum());
  }
  h[kNG - 1]->SetTitle(Form(";%s;N", xTitle.c_str()));
  h[kNG - 1]->SetMaximum(1.25 * ymax);
  h[kNG - 1]->SetFillColorAlpha(kGreen + 2, 0.15);
  h[kNG - 1]->Draw("HIST");
  for (int s = kNG - 2; s >= 0; --s) h[s]->Draw("HIST SAME");
  TLegend* leg = new TLegend(0.20, 0.64, 0.59, 0.89);
  leg->SetBorderSize(0);
  for (int s = 0; s < kNG; ++s) leg->AddEntry(h[s], Form("%s (%.0f)", kStageLabels[s], h[s]->GetEntries()), "l");
  leg->Draw();
}

static double edgeBackground(TH1D* h, int nEdge) {
  const int nb = h->GetNbinsX();
  const int n  = std::min(nEdge, nb / 2);
  double sum = 0;
  for (int i = 1; i <= n; ++i) sum += h->GetBinContent(i) + h->GetBinContent(nb + 1 - i);
  return sum / (2.0 * n);
}

// ---------------------------------------------------------------- pomocnicze 2D
static void drawPad2D(TH2D* h, const char* emptyTitle) {
  gPad->SetRightMargin(0.14);
  if (!h || h->GetEntries() < 1) {
    TLatex* t = new TLatex(0.5, 0.5, Form("%s: brak", emptyTitle));
    t->SetNDC(); t->SetTextAlign(22); t->SetTextSize(0.06);
    t->Draw();
    return;
  }
  gPad->SetLogz();
  h->SetTitle(Form("%s (N=%.0f, #rho=%.2f)", h->GetTitle(), h->GetEntries(), h->GetCorrelationFactor()));
  h->Draw("COLZ");
  const double lo = std::max(h->GetXaxis()->GetXmin(), h->GetYaxis()->GetXmin());
  const double hi = std::min(h->GetXaxis()->GetXmax(), h->GetYaxis()->GetXmax());
  TLine* ln = new TLine(lo, lo, hi, hi);
  ln->SetLineColor(kRed + 1); ln->SetLineStyle(2); ln->SetLineWidth(2);
  ln->Draw();
}

// bias i rozdzielczosc w funkcji osi MC (E: wzglednie)
static void resGraphs(TH2D* h, bool relative, TGraphErrors*& gBias, TGraphErrors*& gRes, int minEntries = 20) {
  gBias = new TGraphErrors(); gRes = new TGraphErrors();
  for (int ix = 1; ix <= h->GetNbinsX(); ++ix) {
    TH1D* py = h->ProjectionY("_py", ix, ix);
    const double n = py->GetEntries();
    const double x = h->GetXaxis()->GetBinCenter(ix), dx = 0.5 * h->GetXaxis()->GetBinWidth(ix);
    if (n >= minEntries && (!relative || x > 0)) {
      const double sc = relative ? 1.0 / x : 1.0;
      const double mean = py->GetMean(), rms = py->GetRMS();
      const int ip = gBias->GetN();
      gBias->SetPoint(ip, x, (mean - x) * sc); gBias->SetPointError(ip, dx, py->GetMeanError() * sc);
      gRes->SetPoint(ip, x, rms * sc);         gRes->SetPointError(ip, dx, rms * sc / std::sqrt(2.0 * n));
    }
    delete py;
  }
}

static void drawResSet(const char* name, const std::vector<TH2D*>& hs, const std::vector<TString>& labels,
                       bool relative, const char* xtitle, const char* ytb, const char* ytr) {
  TCanvas* c = new TCanvas(name, name, 1200, 500);
  c->Divide(2, 1);
  TMultiGraph* mb = new TMultiGraph(); TMultiGraph* mr = new TMultiGraph();
  TLegend* leg = new TLegend(0.6, 0.65, 0.98, 0.98);
  int col = 1, nDrawn = 0;
  for (size_t i = 0; i < hs.size(); ++i) {
    if (!hs[i] || hs[i]->GetEntries() < 20) continue;
    TGraphErrors *gb, *gr;
    resGraphs(hs[i], relative, gb, gr);
    if (gb->GetN() == 0) continue;
    if (col == 5 || col == 10) ++col;
    for (auto* g : {gb, gr}) { g->SetMarkerStyle(20 + (col % 10)); g->SetMarkerColor(col); g->SetLineColor(col); }
    mb->Add(gb, "P"); mr->Add(gr, "P");
    leg->AddEntry(gb, labels[i], "p");
    ++col; ++nDrawn;
  }
  if (nDrawn > 0) {
    c->cd(1); mb->Draw("A"); mb->GetXaxis()->SetTitle(xtitle); mb->GetYaxis()->SetTitle(ytb); leg->Draw();
    c->cd(2); mr->Draw("A"); mr->GetXaxis()->SetTitle(xtitle); mr->GetYaxis()->SetTitle(ytr);
    c->SaveAs(Form("./plots_2d/%s.png", name));
  }
}

// =============================================================================
void draw_improved(const std::string& in_file = "pi0_improved.root") {
  gStyle->SetOptStat(0);
  gSystem->mkdir("./plots_improved", kTRUE);
  gSystem->mkdir("./plots_2d", kTRUE);
  TFile* f = TFile::Open(in_file.c_str());
  if (!f || f->IsZombie()) { std::cout << "Nie mozna otworzyc " << in_file << "\n"; return; }

  // ---- 1) pi0: MC vs final ----
  TCanvas* c1 = new TCanvas("c1", "pi0 MC vs final", 1200, 900);
  c1->Divide(2, 2);
  const char* pi0Vars[4] = {"pt", "eta", "E", "m"};
  for (int i = 0; i < 4; ++i) {
    c1->cd(i + 1);
    if (i == 3) {
      TH1D* h = (TH1D*)f->Get("finalPi0_m");
      if (!h) { std::cout << "Brak histogramu: finalPi0_m\n"; continue; }
      h->SetLineColor(kRed + 1); h->SetLineWidth(2);
      h->Draw("HIST");
    } else overlay(f, "Pi0", pi0Vars[i]);
  }
  c1->SaveAs("./plots_improved/pi0_compare.png");

  // ---- 2) gamma ----
  TCanvas* c2 = new TCanvas("c2", "gamma MC vs final", 1500, 500);
  c2->Divide(3, 1);
  const char* gVars[3] = {"pt", "eta", "E"};
  for (int i = 0; i < 3; ++i) { c2->cd(i + 1); overlay(f, "Gamma", gVars[i]); }
  c2->SaveAs("./plots_improved/gamma_compare.png");

  // ---- 3) final/MC ----
  TCanvas* c3 = new TCanvas("c3", "final / MC", 1500, 900);
  c3->Divide(3, 2);
  for (int i = 0; i < 3; ++i) { c3->cd(i + 1); drawRatio(f, "Pi0", gVars[i]); }
  for (int i = 0; i < 3; ++i) { c3->cd(i + 4); drawRatio(f, "Gamma", gVars[i]); }
  c3->SaveAs("./plots_improved/ratio.png");

  // ---- 4) pozostale ----
  TCanvas* c4 = new TCanvas("c4", "inne", 1200, 900);
  c4->Divide(2, 2);
  const char* others[4] = {"h_nPot", "potPi0_m", "h_method", "h_methodGood"};
  for (int i = 0; i < 4; ++i) {
    c4->cd(i + 1);
    TH1D* h = (TH1D*)f->Get(others[i]);
    if (!h) { std::cout << "Brak histogramu: " << others[i] << "\n"; continue; }
    h->SetLineWidth(2);
    h->SetFillColorAlpha(kAzure + 1, 0.3);
    h->Draw("HIST");
  }
  c4->cd(3)->SetBottomMargin(0.25);
  c4->cd(4)->SetBottomMargin(0.25);
  c4->SaveAs("./plots_improved/other.png");

  // ---- 5) rozdzielczosci: gaus + stale tlo ----
  TCanvas* c5 = new TCanvas("c5", "residuals", 1200, 900);
  c5->Divide(2, 2);
  const char* resNames[4]  = {"h_pi_E_res", "h_pi_eta_res", "h_gam_E_res", "h_gam_eta_res"};
  const char* resTitles[4] = {"(E_{#pi^{0}_{MC}} - E_{rec}) / E_{MC}", "#eta_{#pi^{0}_{MC}} - #eta_{rec}",
                              "(E_{#gamma_{MC}} - E_{rec}) / E_{MC}", "#eta_{#gamma_{MC}} - #eta_{rec}"};
  const double r = 0.2;
  const int nEdge = 20;
  for (int i = 0; i < 4; ++i) {
    c5->cd(i + 1);
    TH1D* h = (TH1D*)f->Get(resNames[i]);
    if (!h) { std::cout << "Brak histogramu: " << resNames[i] << "\n"; continue; }
    h->SetTitle("");
    h->GetXaxis()->SetTitle(resTitles[i]);
    h->SetLineWidth(2);
    h->SetFillColorAlpha(kAzure + 1, 0.3);
    h->Draw("HIST");
    if (h->GetEntries() < 50) continue;
    const double bkg = edgeBackground(h, nEdge);
    TF1* fit = new TF1(Form("gausbkg_%d", i), "gaus(0)+pol0(3)", -r, r);
    fit->SetParameters(std::max(h->GetMaximum() - bkg, 1.0), h->GetBinCenter(h->GetMaximumBin()), 0.25 * r, bkg);
    fit->SetParLimits(2, 0, r);
    fit->FixParameter(3, bkg);
    h->Fit(fit, "QR0");
    fit->SetLineColor(kRed + 1); fit->SetLineWidth(2);
    fit->Draw("SAME");
    TF1* bg = new TF1(Form("bkg_%d", i), "pol0", h->GetXaxis()->GetXmin(), h->GetXaxis()->GetXmax());
    bg->SetParameter(0, bkg);
    bg->SetLineColor(kGray + 2); bg->SetLineStyle(2);
    bg->Draw("SAME");
    TLatex* tex = new TLatex();
    tex->SetNDC(); tex->SetTextSize(0.03);
    tex->DrawLatex(0.15, 0.85, Form("#mu = %.4f #pm %.4f", fit->GetParameter(1), fit->GetParError(1)));
    tex->DrawLatex(0.15, 0.80, Form("#sigma = %.4f #pm %.4f", fit->GetParameter(2), fit->GetParError(2)));
  }
  c5->SaveAs("./plots_improved/res.png");

  // ---- 6) dR, vertex (z, x i y) ----
  TCanvas* c7 = new TCanvas("c7", "dR i vertex", 1200, 900);
  c7->Divide(2, 2);
  const char* dRNames[3] = {"finalPi0_dR", "finalGamma_dR", "h_vtx_z"};
  for (int i = 0; i < 3; ++i) {
    c7->cd(i + 1);
    TH1D* h = (TH1D*)f->Get(dRNames[i]);
    if (!h) { std::cout << "Brak histogramu: " << dRNames[i] << "\n"; continue; }
    h->SetLineWidth(2);
    h->SetFillColorAlpha(kAzure + 1, 0.3);
    h->Draw("HIST");
    if (i < 2) gPad->SetLogy();
  }
  c7->cd(4);
  {
    TH1D* hx = (TH1D*)f->Get("h_vtx_x");
    TH1D* hy = (TH1D*)f->Get("h_vtx_y");
    if (hx && hy) {
      hx->SetLineColor(kRed + 1);  hx->SetLineWidth(2);
      hy->SetLineColor(kBlue + 1); hy->SetLineWidth(2);
      hx->SetMaximum(1.2 * std::max(hx->GetMaximum(), hy->GetMaximum()));
      hx->SetTitle(";vertex x, y [mm];N");
      hx->Draw("HIST");
      hy->Draw("HIST SAME");
      TLegend* lv = new TLegend(0.70, 0.75, 0.89, 0.89);
      lv->SetBorderSize(0);
      lv->AddEntry(hx, "x", "l");
      lv->AddEntry(hy, "y", "l");
      lv->Draw();
    } else std::cout << "Brak histogramu: h_vtx_x / h_vtx_y\n";
  }
  c7->SaveAs("./plots_improved/dR_vtx.png");

  // ---- 7) kazda metoda osobno ----
  drawEachMethod(f, "pi0", "E",   "E_{#pi^{0}} [GeV]",            "./plots_improved/pi0_methods_E.png");
  drawEachMethod(f, "pi0", "eta", "#eta_{#pi^{0}}",               "./plots_improved/pi0_methods_eta.png");
  drawEachMethod(f, "pi0", "m",   "m_{#gamma#gamma} [GeV]",       "./plots_improved/pi0_methods_m.png");
  drawEachMethod(f, "pi0", "dR",  "dR(final, MC #pi^{0})",        "./plots_improved/pi0_methods_dR.png");
  drawEachMethod(f, "gam", "E",   "E_{#gamma} [GeV]",             "./plots_improved/gamma_methods_E.png");
  drawEachMethod(f, "gam", "eta", "#eta_{#gamma}",                "./plots_improved/gamma_methods_eta.png");
  drawEachMethod(f, "gam", "dR",  "dR(final, nearest MC #gamma)", "./plots_improved/gamma_methods_dR.png");

  // ---- 8) skumulowane ----
  TCanvas* c9 = new TCanvas("c9", "pi0 skumulowane", 1200, 900);
  c9->Divide(2, 2);
  c9->cd(1); drawCumulative(f, "pi0", "E",   "E_{#pi^{0}} [GeV]");
  c9->cd(2); drawCumulative(f, "pi0", "eta", "#eta_{#pi^{0}}");
  c9->cd(3); drawCumulative(f, "pi0", "m",   "m_{#gamma#gamma} [GeV]");
  c9->cd(4); drawCumulative(f, "pi0", "dR",  "dR(final, MC #pi^{0})");
  c9->SaveAs("./plots_improved/pi0_cumulative.png");

  TCanvas* c10 = new TCanvas("c10", "gamma skumulowane", 1500, 500);
  c10->Divide(3, 1);
  c10->cd(1); drawCumulative(f, "gam", "E",   "E_{#gamma} [GeV]");
  c10->cd(2); drawCumulative(f, "gam", "eta", "#eta_{#gamma}");
  c10->cd(3); drawCumulative(f, "gam", "dR",  "dR(final, nearest MC #gamma)");
  c10->SaveAs("./plots_improved/gamma_cumulative.png");

  // ---- 9) good per metoda ----
  TH1D* hWin  = (TH1D*)f->Get("h_method");
  TH1D* hGood = (TH1D*)f->Get("h_methodGood");
  if (hWin && hGood) {
    TCanvas* c11 = new TCanvas("c11", "good per metoda", 1400, 600);
    c11->Divide(2, 1);
    c11->cd(1);
    gPad->SetBottomMargin(0.2);
    hWin->SetTitle("final #pi^{0} per metoda;;#pi^{0}");
    hWin->SetLineColor(kBlue + 1);   hWin->SetLineWidth(2);   hWin->SetFillColorAlpha(kBlue + 1, 0.3);
    hGood->SetLineColor(kGreen + 2); hGood->SetLineWidth(2);  hGood->SetFillColorAlpha(kGreen + 2, 0.5);
    hWin->SetMaximum(1.25 * std::max(hWin->GetMaximum(), hGood->GetMaximum()));
    hWin->Draw("HIST");
    hGood->Draw("HIST SAME");
    TLegend* leg11 = new TLegend(0.55, 0.74, 0.89, 0.89);
    leg11->SetBorderSize(0);
    leg11->AddEntry(hWin,  "w oknie masy", "f");
    leg11->AddEntry(hGood, "good (zgodne z MC)", "f");
    leg11->Draw();

    c11->cd(2);
    gPad->SetBottomMargin(0.2);
    TH1D* hFrac = (TH1D*)hGood->Clone("h_methodFrac");
    hFrac->SetDirectory(0);
    hFrac->Divide(hGood, hWin, 1, 1, "B");
    hFrac->Scale(100.0);
    hFrac->SetTitle("odsetek good w kazdej metodzie;;good / w oknie [%]");
    hFrac->SetMinimum(0);
    hFrac->SetMaximum(115);
    hFrac->SetFillColorAlpha(kOrange + 1, 0.4);
    hFrac->SetLineColor(kOrange + 7);
    hFrac->SetLineWidth(2);
    hFrac->SetMarkerStyle(20);
    hFrac->Draw("HIST");
    hFrac->Draw("E1 SAME");
    TLatex txt;
    txt.SetTextAlign(21);
    txt.SetTextSize(0.032);
    std::cout << "\nMetoda          w oknie    good   good/w oknie\n";
    for (int i = 1; i <= hFrac->GetNbinsX(); ++i) {
      const double nWin = hWin->GetBinContent(i), nGood = hGood->GetBinContent(i);
      const double pct = hFrac->GetBinContent(i);
      if (nWin <= 0) continue;
      txt.DrawLatex(hFrac->GetBinCenter(i), pct + 3 + hFrac->GetBinError(i), Form("%.1f%%", pct));
      std::cout << Form("%-14s %8.0f %8.0f   %6.1f%%\n", hWin->GetXaxis()->GetBinLabel(i), nWin, nGood, pct);
    }
    c11->SaveAs("./plots_improved/method_good.png");
  } else std::cout << "Brak h_method / h_methodGood w pliku\n";

  // ---- 10) klastry w evencie ----
  TCanvas* c12 = new TCanvas("c12", "klastry w evencie", 1500, 900);
  c12->Divide(3, 2);
  const char* cntNames[5] = {"h_nImproved", "h_nImprovedTrk", "h_nImpTrue", "h_nNewCl", "h_nVtx"};
  const int cntColors[5]  = {kAzure + 1, kOrange + 7, kGreen + 2, kMagenta + 1, kGray + 2};
  for (int i = 0; i < 5; ++i) {
    c12->cd(i + 1);
    TH1D* h = (TH1D*)f->Get(cntNames[i]);
    if (!h) { std::cout << "Brak histogramu: " << cntNames[i] << "\n"; continue; }
    h->SetLineWidth(2);
    h->SetLineColor(cntColors[i]);
    h->SetFillColorAlpha(cntColors[i], 0.3);
    h->SetMinimum(0.5);
    h->Draw("HIST");
    gPad->SetLogy();
    std::cout << Form("%-16s srednia = %.2f\n", cntNames[i], h->GetMean());
  }
  c12->SaveAs("./plots_improved/clusters_per_event.png");

  // ---- 11) nCand (5 grup) ----
  // uwaga: h_nCand_gN jest wypelniany tylko w zdarzeniach, w ktorych grupa N sie uruchomila
  //        (grupa 2 tylko przy dokladnie 1 potential gamma; grupy 3-5 tylko gdy poprzednie zawiodly)
  TCanvas* c13 = new TCanvas("c13", "liczba kombinacji", 1500, 900);
  c13->Divide(3, 2);
  for (int g = 0; g < kNG; ++g) {
    c13->cd(g + 1);
    TH1D* h = (TH1D*)f->Get(Form("h_nCand_g%d", g + 1));
    if (!h) { std::cout << "Brak histogramu: h_nCand_g" << g + 1 << "\n"; continue; }
    h->SetLineWidth(2);
    h->SetLineColor(kStageColors[g]);
    h->SetFillColorAlpha(kStageColors[g], 0.2);
    h->SetMinimum(0.5);
    h->Draw("HIST");
    gPad->SetLogy();
    std::cout << Form("grupa %d: zdarzen (grupa uruchomiona) %.0f, srednia liczba kombinacji = %.1f\n",
                      g + 1, h->GetEntries(), h->GetMean());
  }
  c13->SaveAs("./plots_improved/nCand.png");

  // ---- 12) pochodzenie klastrow ----
  TH1D* hImp    = (TH1D*)f->Get("h_impCat");
  TH1D* hImpTrk = (TH1D*)f->Get("h_impCatTrk");
  TH1D* hPot    = (TH1D*)f->Get("h_potCat");
  if (hImp && hImpTrk && hPot) {
    TCanvas* c14 = new TCanvas("c14", "pochodzenie klastrow", 1400, 900);
    c14->Divide(2, 2);
    TH1D* catH[3]       = {hImp, hImpTrk, hPot};
    const int catCol[3] = {kAzure + 1, kOrange + 7, kGreen + 2};
    for (int i = 0; i < 3; ++i) {
      c14->cd(i + 1);
      gPad->SetBottomMargin(0.22);
      catH[i]->SetLineWidth(2);
      catH[i]->SetLineColor(catCol[i]);
      catH[i]->SetFillColorAlpha(catCol[i], 0.4);
      catH[i]->SetMarkerSize(1.3);
      catH[i]->SetMaximum(1.2 * catH[i]->GetMaximum());
      catH[i]->Draw("HIST TEXT0");
    }
    c14->cd(4);
    gPad->SetBottomMargin(0.22);
    TH1D* nrm[3];
    const char* nrmLabel[3] = {"improved (zostawione)", "improved (odrzucone: slad)", "klaster potential #gamma"};
    TLegend* leg14 = new TLegend(0.45, 0.70, 0.89, 0.89);
    leg14->SetBorderSize(0);
    double ymax = 0;
    for (int i = 0; i < 3; ++i) {
      nrm[i] = (TH1D*)catH[i]->Clone(Form("nrm_cat_%d", i));
      nrm[i]->SetDirectory(0);
      nrm[i]->SetFillStyle(0);
      nrm[i]->SetMarkerSize(1.0);
      if (nrm[i]->Integral() > 0) nrm[i]->Scale(1.0 / nrm[i]->Integral());
      ymax = std::max(ymax, nrm[i]->GetMaximum());
    }
    nrm[0]->SetTitle("udzial kategorii (znormalizowane);;udzial");
    nrm[0]->SetMaximum(1.25 * ymax);
    for (int i = 0; i < 3; ++i) {
      nrm[i]->Draw(i == 0 ? "HIST" : "HIST SAME");
      leg14->AddEntry(nrm[i], Form("%s (%.0f)", nrmLabel[i], catH[i]->Integral()), "l");
    }
    leg14->Draw();
    c14->SaveAs("./plots_improved/cluster_origin.png");
    for (int i = 0; i < 3; ++i) {
      const double tot = catH[i]->Integral();
      // bin 1 = gamma z pi0 (kategoria 0)
      std::cout << Form("%-28s: %.0f klastrow, w tym gamma z pi0: %.1f%%\n", nrmLabel[i], tot,
                        tot > 0 ? 100.0 * catH[i]->GetBinContent(1) / tot : 0.0);
    }
  } else std::cout << "Brak h_impCat / h_impCatTrk / h_potCat w pliku\n";

  // ======================= 2D: fotony klasa x technologia =======================
  for (int v = 0; v < 3; ++v) {
    TCanvas* c = new TCanvas(Form("cg_%s", varTag[v]), varTag[v], 2400, 1800);
    c->Divide(kNTech, kNCls, 0.001, 0.001);
    std::cout << "\nFotony, zmienna " << varTag[v] << ": liczba wpisow / wspolczynnik korelacji\n";
    printf("  %-4s", "");
    for (int t = 0; t < kNTech; ++t) printf(" %16s", techTag[t]);
    printf("\n");
    for (int cl = 0; cl < kNCls; ++cl) {
      printf("  %-4s", clsTag[cl]);
      for (int t = 0; t < kNTech; ++t) {
        c->cd(cl * kNTech + t + 1);
        TH2D* h = (TH2D*)f->Get(Form("h2_gam_%s_%s_%s", varTag[v], clsTag[cl], techTag[t]));
        drawPad2D(h, Form("%s, %s", clsTag[cl], techTag[t]));
        if (h && h->GetEntries() > 0) printf(" %8.0f / %5.2f", h->GetEntries(), h->GetCorrelationFactor());
        else                          printf(" %16s", "-");
      }
      printf("\n");
    }
    c->SaveAs(Form("./plots_2d/gamma_%s.png", varTag[v]));
  }

  // ---- 2D pi0: zmienna x grupa (5 grup) ----
  TCanvas* cp = new TCanvas("cp", "pi0 2D", 2500, 1500);
  cp->Divide(kNG, 3, 0.001, 0.001);
  std::cout << "\npi0: liczba wpisow / wspolczynnik korelacji\n";
  printf("  %-4s", "");
  for (int g = 0; g < kNG; ++g) printf(" %16s", grpTag[g]);
  printf("\n");
  for (int v = 0; v < 3; ++v) {
    printf("  %-4s", varTag[v]);
    for (int g = 0; g < kNG; ++g) {
      cp->cd(v * kNG + g + 1);
      TH2D* h = (TH2D*)f->Get(Form("h2_pi0_%s_%s", varTag[v], grpTag[g]));
      drawPad2D(h, Form("pi0 %s", grpTag[g]));
      if (h && h->GetEntries() > 0) printf(" %8.0f / %5.2f", h->GetEntries(), h->GetCorrelationFactor());
      else                          printf(" %16s", "-");
    }
    printf("\n");
  }
  cp->SaveAs("./plots_2d/pi0_2d.png");

  // ======================= bias / rozdzielczosc w funkcji MC =======================
  const bool  rel[3] = {true, false, false};
  const char* xt[3]  = {"E_{MC} [GeV]", "#eta_{MC}", "#phi_{MC} [rad]"};
  const char* yb[3]  = {"(<E_{rec}>-E_{MC})/E_{MC}", "<#eta_{rec}>-#eta_{MC}", "<#phi_{rec}>-#phi_{MC}"};
  const char* yr[3]  = {"#sigma(E_{rec})/E_{MC}", "#sigma(#eta_{rec})", "#sigma(#phi_{rec})"};
  for (int v = 0; v < 3; ++v) {
    std::vector<TH2D*> hs; std::vector<TString> lb;
    for (int g = 0; g < kNG; ++g) {
      hs.push_back((TH2D*)f->Get(Form("h2_pi0_%s_%s", varTag[v], grpTag[g])));
      lb.push_back(grpLab[g]);
    }
    drawResSet(Form("res_pi0_%s", varTag[v]), hs, lb, rel[v], xt[v], yb[v], yr[v]);
  }
  for (int v = 0; v < 3; ++v)
    for (int cl = 0; cl < kNCls; ++cl) {
      std::vector<TH2D*> hs; std::vector<TString> lb;
      for (int t = 0; t < kNTech; ++t) {
        hs.push_back((TH2D*)f->Get(Form("h2_gam_%s_%s_%s", varTag[v], clsTag[cl], techTag[t])));
        lb.push_back(techTag[t]);
      }
      drawResSet(Form("res_gam_%s_%s", varTag[v], clsTag[cl]), hs, lb, rel[v], xt[v], yb[v], yr[v]);
    }
}
