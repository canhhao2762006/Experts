#property strict
#include <Trade/Trade.mqh>
CTrade trade;

//==================== INPUTS ====================
input int    MagicNumber            = 26022026;

// Step settings
input int    MaxStepsPerRound       = 10;      // tối đa step trong 1 vòng
input double TriggerPrice           = 0.5;     // giá đi ngược >= TriggerPrice thì mở step mới (đảo chiều)

// Lot settings
input double BaseLots               = 0.01;    // lot step 1
input double LotIncreasePerStep     = 0.01;    // mỗi step +0.01 lot

// Close-all condition
input double CloseAllProfitUSD      = 1.0;     // tổng profit > 1$ => đóng tất cả và reset vòng

// Execution
input int    SlippagePoints         = 30;
input double MaxSpreadPoints        = 0;       // 0 = không lọc spread

// Daily stop (giữ theo yêu cầu cũ: có thể tắt bằng cách set rất lớn)
input double DailyProfitTargetUSD   = 20.0;
input double DailyLossLimitUSD      = 20.0;

// If max step reached
input bool   StopForDayOnMaxStep    = true;    // chạm max step => dừng trong ngày
input bool   CloseAllOnStop         = true;    // dừng thì đóng tất cả lệnh

// Anti-frozen / retry
input int    ReopenDelayMs          = 1500;    // delay trước khi mở lệnh mới
input int    RetryDelayMs           = 1200;    // delay giữa các lần retry
input int    MaxRetries             = 10;      // số lần retry tối đa

//==================== STATE ====================
int      g_step = 1;
int      g_dir  = +1;        // +1 BUY, -1 SELL (hướng của step tiếp theo)
double   g_lastEntryPrice = 0.0;
int      g_lastDir = 0;      // hướng lệnh vừa mở cuối cùng

bool     g_tradingEnabled = true;

double   g_dayStartEquity = 0.0;
int      g_dayKey = 0;

// pending open state (to avoid frozen)
bool   g_pendingOpen = false;
int    g_pendingDir  = 0;
uint   g_dueMs       = 0;
int    g_retryCount  = 0;

// for manual close detection
int    g_prevPosCount = 0;

//==================== HELPERS ====================
int DayKey(datetime t)
{
   MqlDateTime dt; TimeToStruct(t, dt);
   return dt.year*10000 + dt.mon*100 + dt.day;
}

double DayPnL()
{
   return AccountInfoDouble(ACCOUNT_EQUITY) - g_dayStartEquity;
}

bool SpreadOK()
{
   if(MaxSpreadPoints <= 0) return true;
   double sp = (double)SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
   return (sp > 0 && sp <= MaxSpreadPoints);
}

double NormalizeLots(double lots)
{
   double minv = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxv = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   if(step <= 0) step = 0.01;

   lots = MathRound(lots/step) * step;
   if(lots < minv) lots = minv;
   if(lots > maxv) lots = maxv;
   return lots;
}

double LotsForStep(int step)
{
   return NormalizeLots(BaseLots + (double)(step - 1) * LotIncreasePerStep);
}

int OurPositionsCount()
{
   int cnt = 0;
   for(int i=PositionsTotal()-1; i>=0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(!PositionSelectByTicket(ticket)) continue;

      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if((int)PositionGetInteger(POSITION_MAGIC) != MagicNumber) continue;

      cnt++;
   }
   return cnt;
}

double OurPositionsProfitUSD()
{
   // Tổng profit (floating) của tất cả lệnh bot (profit + swap).
   // Commission không có trực tiếp theo position, nên bỏ qua.
   double sum = 0.0;
   for(int i=PositionsTotal()-1; i>=0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(!PositionSelectByTicket(ticket)) continue;

      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if((int)PositionGetInteger(POSITION_MAGIC) != MagicNumber) continue;

      sum += PositionGetDouble(POSITION_PROFIT);
      sum += PositionGetDouble(POSITION_SWAP);
   }
   return sum;
}

bool CloseAllOurPositions()
{
   bool allOk = true;
   // Close from last to first
   for(int i=PositionsTotal()-1; i>=0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(!PositionSelectByTicket(ticket)) continue;

      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if((int)PositionGetInteger(POSITION_MAGIC) != MagicNumber) continue;

      if(!trade.PositionClose(ticket))
      {
         allOk = false;
         Print("CLOSE FAIL ticket=", ticket,
               " retcode=", trade.ResultRetcode(),
               " desc=", trade.ResultRetcodeDescription());
      }
   }
   return allOk;
}

void ResetDailyIfNeeded()
{
   int dk = DayKey(TimeCurrent());
   if(dk != g_dayKey)
   {
      g_dayKey = dk;
      g_dayStartEquity = AccountInfoDouble(ACCOUNT_EQUITY);

      // reset vòng mỗi ngày
      g_step = 1;
      g_dir  = +1;
      g_lastEntryPrice = 0.0;
      g_lastDir = 0;

      g_tradingEnabled = true;

      g_pendingOpen = false;
      g_retryCount = 0;

      g_prevPosCount = 0;
   }
}

void StopForDay()
{
   g_tradingEnabled = false;
   g_pendingOpen = false;

   if(CloseAllOnStop)
      CloseAllOurPositions();
}

void ScheduleOpen(int dir, int delayMs)
{
   if(!g_tradingEnabled) return;
   g_pendingOpen = true;
   g_pendingDir  = dir;
   g_dueMs       = GetTickCount() + (uint)MathMax(0, delayMs);
   g_retryCount  = 0;
}

bool TryOpenNow(int dir)
{
   if(!g_tradingEnabled) return false;
   if(!SpreadOK()) return false;

   double lots = LotsForStep(g_step);
   if(lots <= 0) return false;

   trade.SetExpertMagicNumber(MagicNumber);
   trade.SetDeviationInPoints(SlippagePoints);

   double entry = (dir > 0) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK)
                            : SymbolInfoDouble(_Symbol, SYMBOL_BID);

   bool ok = (dir > 0)
             ? trade.Buy(lots, _Symbol, entry, 0.0, 0.0, "STEP_BUY_NO_SL")
             : trade.Sell(lots, _Symbol, entry, 0.0, 0.0, "STEP_SELL_NO_SL");

   if(!ok)
   {
      Print("OPEN FAIL step=", g_step,
            " lots=", DoubleToString(lots, 2),
            " retcode=", trade.ResultRetcode(),
            " desc=", trade.ResultRetcodeDescription());
      return false;
   }

   // Update last opened info
   g_lastEntryPrice = entry;
   g_lastDir = dir;

   Print("OPEN OK step=", g_step,
         " lots=", DoubleToString(lots, 2),
         " dir=", (dir>0?"BUY":"SELL"),
         " entry=", DoubleToString(entry, _Digits));

   return true;
}

bool IsRetryableRetcode(uint rc)
{
   // Always-available constants:
   if(rc == TRADE_RETCODE_FROZEN) return true;
   if(rc == TRADE_RETCODE_REQUOTE) return true;
   if(rc == TRADE_RETCODE_PRICE_CHANGED) return true;

   // Fallback numeric retcodes commonly seen:
   // 136 OFF_QUOTES, 137 BROKER_BUSY, 146 CONTEXT_BUSY (often)
   if(rc == 136 || rc == 137 || rc == 146) return true;

   return false;
}

void AdvanceStepAndFlip()
{
   g_step++;

   if(g_step > MaxStepsPerRound)
   {
      if(StopForDayOnMaxStep)
      {
         StopForDay();
         return;
      }
      // Nếu không stop thì reset vòng
      g_step = 1;
   }

   // đảo chiều step mới
   g_dir = -g_dir;
}

//==================== CORE LOGIC ====================
void Manage()
{
   ResetDailyIfNeeded();

   // Daily stop
   double dayPnl = DayPnL();
   if(dayPnl >= DailyProfitTargetUSD) { StopForDay(); return; }
   if(dayPnl <= -DailyLossLimitUSD)   { StopForDay(); return; }

   if(!g_tradingEnabled) return;

   // Manual close detection: nếu user đóng hết lệnh bot bằng tay -> reset step 1
   int posCount = OurPositionsCount();
   if(g_prevPosCount > 0 && posCount == 0)
   {
      g_step = 1;
      g_dir = +1;
      g_lastEntryPrice = 0.0;
      g_lastDir = 0;
   }
   g_prevPosCount = posCount;

   // Close-all when total profit > 1 USD
   double totalProfit = OurPositionsProfitUSD();
   if(totalProfit > CloseAllProfitUSD)
   {
      CloseAllOurPositions();
      // reset vòng
      g_step = 1;
      g_dir  = +1;
      g_lastEntryPrice = 0.0;
      g_lastDir = 0;

      // mở vòng mới
      ScheduleOpen(g_dir, ReopenDelayMs);
      return;
   }

   // If no positions, ensure start
   if(posCount == 0)
   {
      if(!g_pendingOpen)
         ScheduleOpen(g_dir, 0);
      return;
   }

   // Trigger next step based on LAST OPENED trade entry price & direction
   if(g_lastDir == 0 || g_lastEntryPrice == 0.0)
   {
      // If we don't know last, don't trigger; wait.
      return;
   }

   double priceNow = (g_lastDir > 0) ? SymbolInfoDouble(_Symbol, SYMBOL_BID)
                                     : SymbolInfoDouble(_Symbol, SYMBOL_ASK);

   double unfavorableMove = (g_lastDir > 0)
                           ? (g_lastEntryPrice - priceNow)   // buy: price down => unfavorable
                           : (priceNow - g_lastEntryPrice);  // sell: price up => unfavorable

   if(unfavorableMove >= TriggerPrice)
   {
      // mở step mới (giữ lệnh cũ), tăng lot + đảo chiều
      AdvanceStepAndFlip();
      if(g_tradingEnabled)
         ScheduleOpen(g_dir, ReopenDelayMs);
      return;
   }
}

//==================== EVENTS ====================
int OnInit()
{
   trade.SetExpertMagicNumber(MagicNumber);
   trade.SetDeviationInPoints(SlippagePoints);

   g_dayKey = DayKey(TimeCurrent());
   g_dayStartEquity = AccountInfoDouble(ACCOUNT_EQUITY);

   g_step = 1;
   g_dir  = +1;
   g_tradingEnabled = true;

   g_pendingOpen = false;
   g_retryCount = 0;
   g_prevPosCount = OurPositionsCount();

   EventSetTimer(1);
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   EventKillTimer();
}

void OnTick()
{
   Manage();
}

void OnTimer()
{
   if(!g_tradingEnabled || !g_pendingOpen) return;
   if(GetTickCount() < g_dueMs) return;

   bool ok = TryOpenNow(g_pendingDir);
   if(ok)
   {
      g_pendingOpen = false;
      g_retryCount = 0;
      return;
   }

   uint rc = trade.ResultRetcode();
   if(IsRetryableRetcode(rc) && g_retryCount < MaxRetries)
   {
      g_retryCount++;
      g_dueMs = GetTickCount() + (uint)RetryDelayMs;
      Print("RETRY ", g_retryCount, "/", MaxRetries, " retcode=", rc,
            " (", trade.ResultRetcodeDescription(), ")");
   }
   else
   {
      Print("GIVE UP opening retcode=", rc, " desc=", trade.ResultRetcodeDescription());
      g_pendingOpen = false;
      g_retryCount = 0;
   }
}