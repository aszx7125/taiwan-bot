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
      home: const MainDashboard(), // 改從這個主儀表板啟動
    );
  }
}

// ==========================================
// 🏠 這是 App 的主外殼 (包含底部導覽列)
// ==========================================
class MainDashboard extends StatefulWidget {
  const MainDashboard({super.key});

  @override
  State<MainDashboard> createState() => _MainDashboardState();
}

class _MainDashboardState extends State<MainDashboard> {
  int _currentIndex = 1; // 預設打開中間的「掃描」分頁

  // 這裡定義三個分頁的畫面
  final List<Widget> _pages = [
    const MarketScreen(), // 左邊：大盤與 TOP 20
    const ScanScreen(),   // 中間：我們剛剛寫好的單股掃描
    const SettingsScreen(),// 右邊：設定與控制
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: _pages[_currentIndex], // 根據目前選中的 Index 顯示對應畫面
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _currentIndex,
        onTap: (index) {
          setState(() {
            _currentIndex = index; // 點擊切換分頁
          });
        },
        items: const [
          BottomNavigationBarItem(icon: Icon(Icons.public), label: '大盤戰情'),
          BottomNavigationBarItem(icon: Icon(Icons.radar), label: '單股掃描'),
          BottomNavigationBarItem(icon: Icon(Icons.settings), label: '策略控制'),
        ],
      ),
    );
  }
}

// ==========================================
// 🌍 分頁 1：大盤戰情室 (會自動向 API 抓取 TOP 20)
// ==========================================
class MarketScreen extends StatefulWidget {
  const MarketScreen({super.key});

  @override
  State<MarketScreen> createState() => _MarketScreenState();
}

class _MarketScreenState extends State<MarketScreen> {
  List<dynamic> _top20List = [];
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _fetchTop20(); // 畫面一載入就自動去抓資料
  }

  String get apiUrl {
    // 使用 Render 部署的公開 API 網址
    const base = 'https://taiwan-bot.onrender.com';
    return '$base/api/v1/market/top20';
  }

  Future<void> _fetchTop20() async {
    try {
      final response = await http.get(Uri.parse(apiUrl));
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        setState(() {
          _top20List = data['top20'];
          _isLoading = false;
        });
      }
    } catch (e) {
      setState(() { _isLoading = false; });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('🌍 大盤與 TOP 20', style: TextStyle(fontWeight: FontWeight.bold)),
        elevation: 0,
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator(color: Colors.tealAccent))
          : RefreshIndicator(
              color: Colors.tealAccent,
              onRefresh: _fetchTop20, // 支援下拉更新！
              child: ListView.builder(
                padding: const EdgeInsets.all(16),
                itemCount: _top20List.length + 1, // +1 是為了把標題算進去
                itemBuilder: (context, index) {
                  if (index == 0) {
                    return const Padding(
                      padding: EdgeInsets.only(bottom: 16),
                      child: Text('🎯 AI 多頭推薦 TOP 20', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.white)),
                    );
                  }
                  
                  final stock = _top20List[index - 1];
                  final bool isStrong = stock['win_prob'] >= 60.0;
                  final Color cardColor = isStrong ? const Color(0xFF00E676) : const Color(0xFF69F0AE);

                  return Container(
                    margin: const EdgeInsets.only(bottom: 12),
                    decoration: BoxDecoration(
                      color: const Color(0xFF1E1E1E),
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: cardColor.withOpacity(0.3)),
                    ),
                    child: ListTile(
                      contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                      leading: CircleAvatar(
                        backgroundColor: cardColor.withOpacity(0.2),
                        child: Text('${index}', style: TextStyle(color: cardColor, fontWeight: FontWeight.bold)),
                      ),
                      title: Text('${stock['ticker']} ${stock['name']}', style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 18, color: Colors.white)),
                      subtitle: Text('多頭勝率: ${stock['win_prob']}% | 現價: \$${stock['current_price']}', style: const TextStyle(color: Colors.grey)),
                      trailing: Icon(isStrong ? Icons.local_fire_department : Icons.trending_up, color: cardColor),
                    ),
                  );
                },
              ),
            ),
    );
  }
}

// ==========================================
// ⚙️ 分頁 3：策略與控制 (預留給你接 GitHub 爬蟲)
// ==========================================
class SettingsScreen extends StatelessWidget {
  const SettingsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('⚙️ 策略控制', style: TextStyle(fontWeight: FontWeight.bold))),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const Text('🛠️ 遠端自動化控制', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.white)),
            const SizedBox(height: 16),
            ElevatedButton.icon(
              onPressed: () {},
              icon: const Icon(Icons.rocket_launch),
              label: const Text('啟動全市場 AI 掃描'),
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFF2A2A2A),
                foregroundColor: Colors.white,
                padding: const EdgeInsets.all(16),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// ==========================================
// ⚡ 分頁 2：單股診斷 (這就是我們剛剛寫好的功能，原封不動搬進來)
// ==========================================
class ScanScreen extends StatefulWidget {
  const ScanScreen({super.key});

  @override
  State<ScanScreen> createState() => _ScanScreenState();
}

class _ScanScreenState extends State<ScanScreen> {
  Map<String, dynamic>? _stockData;
  bool _isLoading = false;
  String _errorMessage = '';
  final TextEditingController _tickerController = TextEditingController(text: '2330');

  String get apiUrl {
    final ticker = _tickerController.text.trim();
    // 使用 Render 部署的公開 API 網址
    const base = 'https://taiwan-bot.onrender.com';
    return '$base/api/v1/scan/$ticker';
  }

  Future<void> fetchAiData() async {
    if (_tickerController.text.isEmpty) return;
    setState(() { _isLoading = true; _errorMessage = ''; _stockData = null; });
    try {
      final response = await http.get(Uri.parse(apiUrl));
      if (response.statusCode == 200) {
        setState(() { _stockData = jsonDecode(response.body); });
      } else {
        setState(() { _errorMessage = "連線失敗，狀態碼: ${response.statusCode}"; });
      }
    } catch (e) {
      setState(() { _errorMessage = "發生錯誤：無法連線到 Python 伺服器。"; });
    } finally {
      setState(() { _isLoading = false; });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('⚡ 單股掃描', style: TextStyle(fontWeight: FontWeight.bold)), elevation: 0),
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
                      hintText: '輸入股票代號 (例: 2330)',
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
            const SizedBox(height: 24),
            Expanded(child: _buildResultArea()),
          ],
        ),
      ),
    );
  }

  Widget _buildResultArea() {
    if (_isLoading) return const Center(child: CircularProgressIndicator(color: Colors.tealAccent));
    if (_errorMessage.isNotEmpty) return Center(child: Text(_errorMessage, style: const TextStyle(color: Colors.redAccent, fontSize: 16), textAlign: TextAlign.center));
    if (_stockData == null) return const Center(child: Text('輸入股票代號開始分析', style: TextStyle(color: Colors.grey, fontSize: 16)));

    final ai = _stockData!['ai_analysis'];
    final details = ai['details'];
    final signal = ai['signal'];
    final double bestLong = ai['best_long_prob'];
    final double bestShort = ai['best_short_prob'];

    Color cardColor = Colors.grey;
    String signalText = "⚪ 動能不足，建議觀望";
    if (signal == "STRONG_LONG") { cardColor = const Color(0xFF00E676); signalText = "⭐⭐⭐ 強勢做多"; }
    else if (signal == "LONG") { cardColor = const Color(0xFF69F0AE); signalText = "⭐⭐ 偏多操作"; }
    else if (signal == "STRONG_SHORT") { cardColor = const Color(0xFFFF1744); signalText = "⚠️⚠️ 強勢放空"; }
    else if (signal == "SHORT") { cardColor = const Color(0xFFFF5252); signalText = "⚠️ 偏空操作"; }
    else if (signal == "HIGH_VOLATILITY") { cardColor = Colors.orangeAccent; signalText = "⚡ 多空雙巴"; }

    return SingleChildScrollView(
      child: Container(
        decoration: BoxDecoration(color: const Color(0xFF1E1E1E), borderRadius: BorderRadius.circular(16), border: Border.all(color: cardColor.withOpacity(0.5), width: 2)),
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('⚔️ 對撞結果：$signalText', style: TextStyle(color: cardColor, fontSize: 16, fontWeight: FontWeight.bold)),
            const SizedBox(height: 8),
            Text('${_stockData!['ticker']} ${_stockData!['name']}', style: const TextStyle(fontSize: 24, fontWeight: FontWeight.bold, color: Colors.white)),
            const Divider(color: Colors.white24, height: 30),
            Row(
              children: [
                Expanded(child: Container(padding: const EdgeInsets.all(12), decoration: BoxDecoration(color: const Color(0xFF00E676).withOpacity(0.1), borderRadius: BorderRadius.circular(12)), child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [const Text('🟢 多頭極值', style: TextStyle(color: Color(0xFF00E676), fontWeight: FontWeight.bold)), const SizedBox(height: 4), Text('${bestLong.toStringAsFixed(1)}%', style: const TextStyle(fontSize: 28, fontWeight: FontWeight.bold, color: Color(0xFF00E676))), const SizedBox(height: 8), Text('▶ LGBM: ${details['lgbm_long']}%', style: const TextStyle(color: Colors.white70, fontSize: 12)), Text('▶ LSTM: ${details['lstm_long']}%', style: const TextStyle(color: Colors.white70, fontSize: 12))]))),
                const SizedBox(width: 12),
                Expanded(child: Container(padding: const EdgeInsets.all(12), decoration: BoxDecoration(color: const Color(0xFFFF1744).withOpacity(0.1), borderRadius: BorderRadius.circular(12)), child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [const Text('🔴 空頭極值', style: TextStyle(color: Color(0xFFFF1744), fontWeight: FontWeight.bold)), const SizedBox(height: 4), Text('${bestShort.toStringAsFixed(1)}%', style: const TextStyle(fontSize: 28, fontWeight: FontWeight.bold, color: Color(0xFFFF1744))), const SizedBox(height: 8), Text('▶ LGBM: ${details['lgbm_short']}%', style: const TextStyle(color: Colors.white70, fontSize: 12)), Text('▶ LSTM: ${details['lstm_short']}%', style: const TextStyle(color: Colors.white70, fontSize: 12))]))),
              ],
            ),
            const Divider(color: Colors.white24, height: 40),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                _buildPriceColumn('現價', _stockData!['current_price'], Colors.white),
                _buildPriceColumn('上檔壓力', _stockData!['res_level'], Colors.orangeAccent),
                _buildPriceColumn('下檔支撐', _stockData!['sup_level'], Colors.lightBlueAccent),
              ],
            )
          ],
        ),
      ),
    );
  }

  Widget _buildPriceColumn(String label, dynamic price, Color color) {
    return Column(children: [Text(label, style: const TextStyle(color: Colors.grey, fontSize: 12)), const SizedBox(height: 4), Text(price.toString(), style: TextStyle(color: color, fontSize: 18, fontWeight: FontWeight.bold))]);
  }
}