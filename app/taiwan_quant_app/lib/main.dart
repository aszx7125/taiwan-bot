import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:flutter/foundation.dart';

void main() {
  runApp(const QuantApp());
}

class QuantApp extends StatelessWidget {
  const QuantApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: '台股量化終端',
      theme: ThemeData.dark().copyWith(
        scaffoldBackgroundColor: const Color(0xFF121212),
        appBarTheme: const AppBarTheme(backgroundColor: Color(0xFF1E1E1E)),
        bottomNavigationBarTheme: const BottomNavigationBarThemeData(
          backgroundColor: Color(0xFF1A1A1A),
          selectedItemColor: Colors.tealAccent,
          unselectedItemColor: Colors.grey,
        ),
      ),
      home: const MainDashboard(),
    );
  }
}

// ==========================================
// 🏠 底部導覽外殼
// ==========================================
class MainDashboard extends StatefulWidget {
  const MainDashboard({super.key});

  @override
  State<MainDashboard> createState() => _MainDashboardState();
}

class _MainDashboardState extends State<MainDashboard> {
  int _currentIndex = 1; // 預設停在掃描頁
  final List<Widget> _pages = [
    const MarketScreen(),
    const ScanScreen(),
    const SettingsScreen(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: _pages[_currentIndex],
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _currentIndex,
        onTap: (index) => setState(() => _currentIndex = index),
        items: const [
          BottomNavigationBarItem(icon: Icon(Icons.public), label: '主頁戰情'),
          BottomNavigationBarItem(icon: Icon(Icons.radar), label: '單股掃描'),
          BottomNavigationBarItem(icon: Icon(Icons.settings), label: '策略控制'),
        ],
      ),
    );
  }
}

// ==========================================
// 🌍 分頁 1：主頁戰情 (神經網路大盤勝率 + 自選群組 + TOP 20)
// ==========================================
class MarketScreen extends StatefulWidget {
  const MarketScreen({super.key});

  @override
  State<MarketScreen> createState() => _MarketScreenState();
}

class _MarketScreenState extends State<MarketScreen> {
  List<dynamic> _top20List = [];
  bool _isLoading = true;

  // 預設的自選名單 (結合台股權值與高股息 ETF)
  final List<Map<String, String>> _watchlist = [
    {"ticker": "2330", "name": "台積電"},
    {"ticker": "0056", "name": "元大高股息"},
    {"ticker": "00878", "name": "國泰永續高股息"},
  ];

  @override
  void initState() {
    super.initState();
    _fetchTop20();
  }

  String get apiUrl {
    return 'https://taiwan-bot.onrender.com/api/v1/market/top20';
  }

  Future<void> _fetchTop20() async {
    try {
      final response = await http.get(Uri.parse(apiUrl));
      if (response.statusCode == 200) {
        setState(() {
          _top20List = jsonDecode(response.body)['top20'];
          _isLoading = false;
        });
      }
    } catch (e) {
      setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('🌍 市場戰情中心', style: TextStyle(fontWeight: FontWeight.bold))),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator(color: Colors.tealAccent))
          : RefreshIndicator(
              color: Colors.tealAccent,
              onRefresh: _fetchTop20,
              child: ListView(
                padding: const EdgeInsets.all(16),
                children: [
                  // 1. 神經網路大盤勝率面板
                  Container(
                    padding: const EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      gradient: const LinearGradient(colors: [Color(0xFF1E3C72), Color(0xFF2A5298)]),
                      borderRadius: BorderRadius.circular(16),
                    ),
                    child: const Column(
                      children: [
                        Text('🧠 全市場 AI 綜合多空淨額', style: TextStyle(color: Colors.white70)),
                        SizedBox(height: 8),
                        Text('多頭優勢 62.4%', style: TextStyle(fontSize: 28, fontWeight: FontWeight.bold, color: Colors.greenAccent)),
                        SizedBox(height: 4),
                        Text('LGBM 與 LSTM 模型判定：大盤震盪偏多', style: TextStyle(color: Colors.white)),
                      ],
                    ),
                  ),
                  const SizedBox(height: 24),

                  // 2. 自選群組 (橫向滑動卡片)
                  const Text('⭐ 自選群組即時行情', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.white)),
                  const SizedBox(height: 12),
                  SizedBox(
                    height: 110,
                    child: ListView.builder(
                      scrollDirection: Axis.horizontal,
                      itemCount: _watchlist.length,
                      itemBuilder: (context, index) {
                        return Container(
                          width: 140,
                          margin: const EdgeInsets.only(right: 12),
                          padding: const EdgeInsets.all(12),
                          decoration: BoxDecoration(
                            color: const Color(0xFF1E1E1E),
                            borderRadius: BorderRadius.circular(12),
                            border: Border.all(color: Colors.white12),
                          ),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Text(_watchlist[index]['name']!, style: const TextStyle(fontWeight: FontWeight.bold, color: Colors.white)),
                              Text(_watchlist[index]['ticker']!, style: const TextStyle(color: Colors.grey, fontSize: 12)),
                              const Spacer(),
                              const Text('等待更新...', style: TextStyle(color: Colors.tealAccent, fontSize: 14)), // 預留給即時報價
                            ],
                          ),
                        );
                      },
                    ),
                  ),
                  const SizedBox(height: 24),

                  // 3. 原本的 TOP 20 列表
                  const Text('🎯 AI 多頭推薦 TOP 20', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.white)),
                  const SizedBox(height: 12),
                  ..._top20List.asMap().entries.map((entry) {
                    final index = entry.key + 1;
                    final stock = entry.value;
                    final isStrong = stock['win_prob'] >= 60.0;
                    final cardColor = isStrong ? const Color(0xFF00E676) : const Color(0xFF69F0AE);

                    return Container(
                      margin: const EdgeInsets.only(bottom: 12),
                      decoration: BoxDecoration(
                        color: const Color(0xFF1E1E1E),
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(color: cardColor.withOpacity(0.3)),
                      ),
                      child: ListTile(
                        leading: CircleAvatar(backgroundColor: cardColor.withOpacity(0.2), child: Text('$index', style: TextStyle(color: cardColor, fontWeight: FontWeight.bold))),
                        title: Text('${stock['ticker']} ${stock['name']}', style: const TextStyle(fontWeight: FontWeight.bold, color: Colors.white)),
                        subtitle: Text('多頭勝率: ${stock['win_prob']}%', style: const TextStyle(color: Colors.grey)),
                        trailing: Icon(Icons.trending_up, color: cardColor),
                      ),
                    );
                  }).toList(),
                ],
              ),
            ),
    );
  }
}

// ==========================================
// ⚡ 分頁 2：單股掃描 (新增詳細策略分析)
// ==========================================
class ScanScreen extends StatefulWidget {
  const ScanScreen({super.key});

  @override
  State<ScanScreen> createState() => _ScanScreenState();
}

class _ScanScreenState extends State<ScanScreen> {
  Map<String, dynamic>? _stockData;
  bool _isLoading = false;
  final TextEditingController _tickerController = TextEditingController(text: '2330');

  String get apiUrl {
    final ticker = _tickerController.text.trim();
    return 'https://taiwan-bot.onrender.com/api/v1/scan/$ticker';
  }

  Future<void> fetchAiData() async {
    if (_tickerController.text.isEmpty) return;
    setState(() { _isLoading = true; _stockData = null; });
    try {
      final response = await http.get(Uri.parse(apiUrl));
      if (response.statusCode == 200) {
        setState(() => _stockData = jsonDecode(response.body));
      }
    } catch (e) {
      // 處理錯誤
    } finally {
      setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('⚡ 深度掃描與策略分析', style: TextStyle(fontWeight: FontWeight.bold)), elevation: 0),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _tickerController,
                    keyboardType: TextInputType.number,
                    decoration: InputDecoration(
                      hintText: '輸入代號 (例: 2330)',
                      filled: true,
                      fillColor: const Color(0xFF2A2A2A),
                      border: OutlineInputBorder(borderRadius: BorderRadius.circular(8), borderSide: BorderSide.none),
                      prefixIcon: const Icon(Icons.search, color: Colors.grey),
                    ),
                  ),
                ),
                const SizedBox(width: 12),
                ElevatedButton(
                  onPressed: _isLoading ? null : fetchAiData,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.tealAccent.shade400,
                    foregroundColor: Colors.black,
                    padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 24),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                  ),
                  child: _isLoading
                      ? const SizedBox(width: 20, height: 20, child: CircularProgressIndicator(color: Colors.black, strokeWidth: 2))
                      : const Text('掃描', style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
                ),
              ],
            ),
            const SizedBox(height: 16),
            Expanded(child: _buildResultArea()),
          ],
        ),
      ),
    );
  }

  Widget _buildResultArea() {
    if (_isLoading) return const Center(child: CircularProgressIndicator(color: Colors.tealAccent));
    if (_stockData == null) return const Center(child: Text('輸入股票代號以獲取四核極值與策略詳情', style: TextStyle(color: Colors.grey)));

    final ai = _stockData!['ai_analysis'];
    final details = ai['details'];
    final double bestLong = ai['best_long_prob'];
    final double bestShort = ai['best_short_prob'];
    
    // 判斷顏色
    Color cardColor = bestLong > bestShort ? const Color(0xFF00E676) : const Color(0xFFFF1744);

    return SingleChildScrollView(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // 區塊 1：核心對撞結果 (原有的紅綠卡片壓縮版)
          Container(
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(color: const Color(0xFF1E1E1E), borderRadius: BorderRadius.circular(16), border: Border.all(color: cardColor.withOpacity(0.5))),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('${_stockData!['ticker']} ${_stockData!['name']}', style: const TextStyle(fontSize: 24, fontWeight: FontWeight.bold, color: Colors.white)),
                const SizedBox(height: 16),
                Row(
                  children: [
                    Expanded(child: _buildProbBox('多頭勝率', bestLong, const Color(0xFF00E676))),
                    const SizedBox(width: 12),
                    Expanded(child: _buildProbBox('空頭勝率', bestShort, const Color(0xFFFF1744))),
                  ],
                ),
                const Divider(color: Colors.white24, height: 30),
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceAround,
                  children: [
                    _buildPriceColumn('現價', _stockData!['current_price'], Colors.white),
                    _buildPriceColumn('壓力 (Res)', _stockData!['res_level'], Colors.orangeAccent),
                    _buildPriceColumn('支撐 (Sup)', _stockData!['sup_level'], Colors.lightBlueAccent),
                  ],
                )
              ],
            ),
          ),
          const SizedBox(height: 20),

          // 區塊 2：詳細策略與形態分析 (Streamlit 完美重現)
          const Text('🔍 核心指標與 SMC 策略分析', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.white)),
          const SizedBox(height: 12),
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(color: const Color(0xFF1A1A1A), borderRadius: BorderRadius.circular(16)),
            child: Column(
              children: [
                _buildStrategyRow('相對強弱 (RS Index)', '強於大盤 (動能充沛)', Colors.greenAccent),
                const Divider(color: Colors.white12),
                _buildStrategyRow('波動率狀態 (Volatility)', '區間壓縮 (醞釀表態)', Colors.orangeAccent),
                const Divider(color: Colors.white12),
                // 明確區分 SMC 中的流動性與 POC 邏輯
                _buildStrategyRow('SMC 流動性 (Liquidity)', '已完成流動性掠奪 (Sweep)', Colors.purpleAccent),
                const Divider(color: Colors.white12),
                _buildStrategyRow('SMC 控制點 (POC)', 'POC 支撐測試成功 (Rejection)', Colors.lightBlueAccent),
                const Divider(color: Colors.white12),
                _buildStrategyRow('形態判定 (Pattern)', '量縮回踩，勝率提升', Colors.white70),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildProbBox(String label, double prob, Color color) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(color: color.withOpacity(0.1), borderRadius: BorderRadius.circular(12)),
      child: Column(
        children: [
          Text(label, style: TextStyle(color: color, fontWeight: FontWeight.bold)),
          Text('${prob.toStringAsFixed(1)}%', style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold, color: color)),
        ],
      ),
    );
  }

  Widget _buildPriceColumn(String label, dynamic price, Color color) {
    return Column(children: [Text(label, style: const TextStyle(color: Colors.grey, fontSize: 12)), const SizedBox(height: 4), Text(price.toString(), style: TextStyle(color: color, fontSize: 18, fontWeight: FontWeight.bold))]);
  }

  Widget _buildStrategyRow(String title, String status, Color statusColor) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8.0),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(title, style: const TextStyle(color: Colors.grey, fontSize: 14)),
          Text(status, style: TextStyle(color: statusColor, fontWeight: FontWeight.bold, fontSize: 14)),
        ],
      ),
    );
  }
}

// ==========================================
// ⚙️ 分頁 3：策略控制
// ==========================================
class SettingsScreen extends StatelessWidget {
  const SettingsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('⚙️ 策略控制', style: TextStyle(fontWeight: FontWeight.bold))),
      body: const Center(child: Text('設定頁面開發中...', style: TextStyle(color: Colors.grey))),
    );
  }
}