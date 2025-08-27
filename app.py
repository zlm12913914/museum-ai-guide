import os
import json
import logging
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import openai
from werkzeug.utils import secure_filename
import sqlite3
from pathlib import Path
import tempfile
import base64

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# 配置
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# OpenAI配置 - 使用新的客户端方式
from openai import OpenAI

client = OpenAI(
    api_key=""
)

from flask import send_from_directory

# 添加对根目录下 index.html 的访问支持
@app.route('/index.html')
def serve_index():
    return send_from_directory('.', 'index.html')

# 创建必要的目录
os.makedirs('data', exist_ok=True)
os.makedirs('uploads', exist_ok=True)
os.makedirs('knowledge_base', exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs('audio_cache', exist_ok=True)  # 音频缓存目录

class MuseumGuideSystem:
    def __init__(self):
        self.nodes_file = 'data/artifacts_nodes.json'
        self.relations_file = 'data/spatial_relations.json'
        self.knowledge_base_dir = 'knowledge_base'
        self.audio_cache_dir = 'audio_cache'
        self.init_database()
        self.load_museum_data()
    
    def init_database(self):
        """初始化SQLite数据库"""
        conn = sqlite3.connect('data/museum_guide.db')
        cursor = conn.cursor()
        
        # 创建用户反馈表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_profile TEXT,
                route_rating REAL,
                content_rating REAL,
                service_rating REAL,
                interaction_rating REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 创建访问日志表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS visit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_profile TEXT,
                tour_route TEXT,
                duration INTEGER,
                questions_asked TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 创建音频缓存表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audio_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text_hash TEXT UNIQUE,
                voice_model TEXT,
                file_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def load_museum_data(self):
        """加载博物馆数据"""
        try:
            # 加载节点数据
            if os.path.exists(self.nodes_file):
                with open(self.nodes_file, 'r', encoding='utf-8') as f:
                    self.artifacts_nodes = json.load(f)
            else:
                self.create_sample_nodes()
            
            # 加载关系数据
            if os.path.exists(self.relations_file):
                with open(self.relations_file, 'r', encoding='utf-8') as f:
                    self.spatial_relations = json.load(f)
            else:
                self.create_sample_relations()
            
            # 加载知识库
            self.load_knowledge_base()
            
        except Exception as e:
            logger.error(f"加载博物馆数据失败: {e}")
    
    def create_sample_nodes(self):
        """创建示例节点数据"""
        self.artifacts_nodes = {
            "artifacts": [
                {
                    "id": "item_001",
                    "name": "青铜立人像",
                    "category": "青铜器",
                    "period": "商晚期",
                    "location": "展厅入口右侧",
                    "coordinates": {"x": 10, "y": 5},
                    "importance": "珍品",
                    "popularity": "热门",
                    "keywords": ["三星堆", "青铜", "立人", "祭祀"],
                    "estimated_time": 5
                },
                {
                    "id": "item_002", 
                    "name": "青铜神树",
                    "category": "青铜器",
                    "period": "商晚期",
                    "location": "中央展区",
                    "coordinates": {"x": 20, "y": 15},
                    "importance": "珍品",
                    "popularity": "热门",
                    "keywords": ["三星堆", "神树", "宇宙观", "青铜"],
                    "estimated_time": 8
                },
                {
                    "id": "item_003",
                    "name": "金沙太阳神鸟",
                    "category": "金器",
                    "period": "商周",
                    "location": "精品展区",
                    "coordinates": {"x": 30, "y": 25},
                    "importance": "珍品",
                    "popularity": "热门",
                    "keywords": ["金沙", "太阳神鸟", "文化遗产标志"],
                    "estimated_time": 6
                },
                {
                    "id": "item_004",
                    "name": "彩绘陶罐",
                    "category": "陶器",
                    "period": "新石器时代",
                    "location": "陶器展区",
                    "coordinates": {"x": 15, "y": 30},
                    "importance": "重要",
                    "popularity": "一般",
                    "keywords": ["彩绘", "陶器", "新石器"],
                    "estimated_time": 3
                },
                {
                    "id": "item_005",
                    "name": "青花瓷瓶",
                    "category": "瓷器", 
                    "period": "明代",
                    "location": "瓷器展区",
                    "coordinates": {"x": 25, "y": 35},
                    "importance": "重要",
                    "popularity": "热门",
                    "keywords": ["青花瓷", "明代", "瓷器"],
                    "estimated_time": 4
                }
            ]
        }
        
        with open(self.nodes_file, 'w', encoding='utf-8') as f:
            json.dump(self.artifacts_nodes, f, ensure_ascii=False, indent=2)
    
    def create_sample_relations(self):
        """创建示例空间关系数据"""
        self.spatial_relations = {
            "exhibition_halls": [
                {
                    "id": "ancient_sichuan_hall",
                    "name": "古代四川馆",
                    "entrance": {"x": 0, "y": 0},
                    "exit": {"x": 40, "y": 40},
                    "areas": [
                        {
                            "name": "入口展区",
                            "bounds": {"x1": 0, "y1": 0, "x2": 20, "y2": 10},
                            "artifacts": ["item_001"]
                        },
                        {
                            "name": "中央展区", 
                            "bounds": {"x1": 10, "y1": 10, "x2": 30, "y2": 20},
                            "artifacts": ["item_002"]
                        },
                        {
                            "name": "精品展区",
                            "bounds": {"x1": 20, "y1": 20, "x2": 40, "y2": 30},
                            "artifacts": ["item_003"]
                        },
                        {
                            "name": "陶器展区",
                            "bounds": {"x1": 5, "y1": 25, "x2": 25, "y2": 35},
                            "artifacts": ["item_004"]
                        },
                        {
                            "name": "瓷器展区",
                            "bounds": {"x1": 15, "y1": 30, "x2": 35, "y2": 40},
                            "artifacts": ["item_005"]
                        }
                    ]
                }
            ],
            "paths": [
                {"from": "item_001", "to": "item_002", "distance": 15, "direction": "向前直行"},
                {"from": "item_002", "to": "item_003", "distance": 18, "direction": "右转前行"},
                {"from": "item_003", "to": "item_004", "distance": 20, "direction": "左转前行"},
                {"from": "item_004", "to": "item_005", "distance": 12, "direction": "向右前行"}
            ]
        }
        
        with open(self.relations_file, 'w', encoding='utf-8') as f:
            json.dump(self.spatial_relations, f, ensure_ascii=False, indent=2)
    
    def load_knowledge_base(self):
        """加载知识库文档"""
        self.knowledge_base = {}
        kb_dir = Path(self.knowledge_base_dir)
        
        if not kb_dir.exists():
            self.create_sample_knowledge_base()
        
        for file_path in kb_dir.glob('*.txt'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    artifact_id = file_path.stem
                    self.knowledge_base[artifact_id] = content
            except Exception as e:
                logger.error(f"加载知识库文件 {file_path} 失败: {e}")
    
    def create_sample_knowledge_base(self):
        """创建示例知识库"""
        knowledge_data = {
            "item_001": """青铜立人像

基本信息：
高度：260.8厘米
年代：商晚期（约公元前1200-1000年）
出土地点：四川广汉三星堆遗址
材质：青铜

文物描述：
这尊青铜立人像是三星堆文化的代表性文物，也是目前世界上发现的最高大的商代青铜立人像。立人头戴莲花状头冠，面相威严，双眼突出，鼻梁高挺，阔嘴大耳。身穿左衽长袍，胸前装饰有龙纹，双手呈握物状，赤足立于方台上。

历史价值：
立人像代表了古代蜀国高超的青铜铸造技术，体现了古蜀文明的独特性。其造型风格与中原地区明显不同，展现了古代四川地区独特的文化传统。学者们普遍认为这是一位具有至高权威的蜀王或大巫师的形象。

文化意义：
立人像的发现改写了中国古代文明的格局，证明了在商周时期，四川地区就存在着高度发达的青铜文明。它是古代蜀国政治、宗教、艺术的完美结合体，被誉为"世界青铜器之王"。""",

            "item_002": """青铜神树

基本信息：
高度：396厘米（复原高度）
年代：商晚期（约公元前1200-1000年）
出土地点：四川广汉三星堆遗址
材质：青铜

文物描述：
青铜神树是三星堆遗址出土的最具神秘色彩的文物之一。神树分为三层，每层三枝，共九枝。树枝上立有神鸟，枝头悬挂果实。树干一侧有一条龙缘树而下，整体造型雄奇瑰丽。

神话背景：
神树的造型与中国古代神话中的"扶桑"、"建木"等神树形象高度吻合。《山海经》记载："汤谷上有扶桑，十日所浴，在黑齿北。居水中，有大木，九日居下枝，一日居上枝。"三星堆神树正是这一神话的完美体现。

象征意义：
神树象征着古代蜀人的宇宙观，体现了他们对天地的理解。树上的神鸟代表太阳，九只神鸟对应九个太阳的传说。整个神树构成了一个沟通天地的宇宙模型，反映了古蜀人"天人合一"的哲学思想。

工艺特色：
神树采用分段铸造、铆接组装的工艺，技术极其复杂。其铸造工艺代表了商代青铜器制作的最高水平，展现了古代蜀国工匠的卓越智慧。""",

            "item_003": """金沙太阳神鸟

基本信息：
直径：12.5厘米
厚度：0.02厘米
年代：商末周初（约公元前1200-1000年）
出土地点：四川成都金沙遗址
材质：金

文物描述：
太阳神鸟金饰是一件极其精美的金器，呈圆形，采用透雕工艺制作。图案分为内外两层：内层为一个放射十二芒的太阳，外层为四只飞鸟，首足前后相接，环绕太阳飞翔。

文化符号：
太阳神鸟已成为中国文化遗产的标志，其图案被广泛应用于各种文化符号中。它体现了古代先民对太阳的崇拜和对飞翔的向往，是古代中国宗教信仰和艺术创作的完美结合。

工艺价值：
金饰的制作工艺极其精湛，厚度仅0.02厘米，却能雕刻出如此精细的图案，体现了古代工匠的高超技艺。其设计简洁而富有动感，线条流畅，是古代艺术的杰作。

象征意义：
太阳神鸟反映了古蜀人的宇宙观念，太阳象征光明和生命，神鸟代表着沟通天地的使者。四只神鸟可能代表四季或四方，体现了古人对时空的理解。""",

            "item_004": """彩绘陶罐

基本信息：
高度：32厘米
年代：新石器时代晚期（约公元前3000-2000年）
出土地点：四川地区
材质：陶土

文物描述：
这件彩绘陶罐造型优美，器身饰有精美的几何纹样和动植物图案。彩绘以红、黑色为主，线条流畅，构图和谐。罐身圆润，口沿外撇，具有典型的新石器时代陶器特征。

制作工艺：
陶罐采用手工拉坯成型，经过精细打磨后施以彩绘。烧制温度适中，质地坚实。彩绘颜料为天然矿物颜料，经过数千年仍保持鲜艳的色彩。

文化价值：
彩绘陶罐反映了新石器时代四川地区先民的生活状况和艺术水平。其装饰图案具有浓厚的生活气息，体现了古代先民的审美情趣和精神追求。

历史意义：
这类彩绘陶器是研究新石器时代四川地区文化发展的重要材料，为了解古代先民的生活方式、宗教信仰和艺术成就提供了珍贵的实物资料。""",

            "item_005": """青花瓷瓶

基本信息：
高度：45厘米
年代：明代永乐年间（1403-1424年）
产地：景德镇
材质：瓷

文物描述：
这件青花瓷瓶造型端庄典雅，瓶身修长，颈部细长，腹部丰满。通体以青花装饰，主题纹样为缠枝莲纹，辅以如意云头纹和蕉叶纹。青花发色浓艳，层次分明。

工艺特点：
瓷瓶胎质细腻洁白，釉色莹润如玉。青花用料为进口的"苏麻离青"，发色浓重艳丽，具有浓淡变化的水墨效果。烧制工艺精湛，器形规整，无丝毫变形。

艺术价值：
瓶身纹样布局合理，疏密有致，体现了明代青花瓷器的典型风格。缠枝莲纹寓意富贵吉祥，是明代瓷器常见的装饰题材。整体造型和装饰体现了中国传统陶瓷艺术的精髓。

收藏价值：
明代永乐青花瓷器被誉为青花瓷器的黄金时代，工艺精湛，存世量稀少，具有极高的收藏价值和研究价值。这件瓷瓶代表了明代官窑青花瓷器的最高水准。"""
        }
        
        for artifact_id, content in knowledge_data.items():
            file_path = Path(self.knowledge_base_dir) / f"{artifact_id}.txt"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
    
    def generate_personalized_route(self, user_profile):
        """生成个性化参观路线"""
        try:
            available_items = self.artifacts_nodes['artifacts'].copy()
            
            # 根据兴趣过滤
            if user_profile['interests']:
                filtered_items = []
                for item in available_items:
                    for interest in user_profile['interests']:
                        if (interest in item.get('keywords', []) or 
                            interest in item.get('category', '') or
                            interest in item.get('importance', '') or
                            interest in item.get('popularity', '')):
                            filtered_items.append(item)
                            break
                
                if filtered_items:
                    available_items = filtered_items
            
            # 根据参观时长选择藏品数量
            duration = user_profile['duration']
            if duration == '10分钟':
                max_items = 2
            elif duration == '30分钟':
                max_items = 4
            else:  # 60分钟
                max_items = len(available_items)
            
            # 按重要性和热门程度排序
            available_items.sort(key=lambda x: (
                x.get('importance') == '珍品',
                x.get('popularity') == '热门'
            ), reverse=True)
            
            selected_items = available_items[:max_items]
            
            # 优化路线顺序（简单的距离优化）
            if len(selected_items) > 1:
                selected_items = self.optimize_route(selected_items)
            
            return selected_items
            
        except Exception as e:
            logger.error(f"生成个性化路线失败: {e}")
            return self.artifacts_nodes['artifacts'][:3]  # 返回默认路线
    
    def optimize_route(self, items):
        """优化参观路线顺序"""
        if len(items) <= 1:
            return items
        
        # 简单的贪心算法优化路线
        optimized = [items[0]]  # 从第一个开始
        remaining = items[1:]
        
        while remaining:
            last_item = optimized[-1]
            last_pos = last_item['coordinates']
            
            # 找到距离最近的下一个点
            closest = min(remaining, key=lambda item: 
                (item['coordinates']['x'] - last_pos['x'])**2 + 
                (item['coordinates']['y'] - last_pos['y'])**2
            )
            
            optimized.append(closest)
            remaining.remove(closest)
        
        return optimized
    
    def get_navigation_info(self, from_item_id, to_item_id):
        """获取导航信息"""
        for path in self.spatial_relations['paths']:
            if path['from'] == from_item_id and path['to'] == to_item_id:
                return path
        return None
    
    def process_user_question(self, question, user_profile, current_context):
        """处理用户问题"""
        try:
            # 构建系统提示词
            system_prompt = self.build_system_prompt(user_profile, current_context)
            
            # 准备上下文信息
            context_info = ""
            if current_context.get('current_item'):
                item_id = current_context['current_item']
                if item_id in self.knowledge_base:
                    context_info = f"当前藏品信息：\n{self.knowledge_base[item_id]}\n\n"
            
            # 调用OpenAI API - 使用新的客户端方式
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{context_info}用户问题：{question}"}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"处理用户问题失败: {e}")
            return "抱歉，我暂时无法回答您的问题，请您稍后再试。"
    
    def build_system_prompt(self, user_profile, context):
        """构建系统提示词"""
        age_desc = {
            '10-18': '青少年',
            '18-35': '年轻人',
            '35-50': '中年人',
            '50+': '老年人'
        }.get(user_profile['age'], '参观者')
        
        prompt = f"""您是四川博物院古代四川馆的AI导览员。您正在为一位{user_profile['gender']}性{age_desc}（{user_profile['education']}文化程度）提供服务。

参观者信息：
- 年龄段：{user_profile['age']}
- 文化程度：{user_profile['education']}
- 参观时长：{user_profile['duration']}
- 兴趣爱好：{', '.join(user_profile['interests']) if user_profile['interests'] else '无特定兴趣'}
- 使用语言：{user_profile['language']}

请根据参观者的背景，用适合的语言风格和深度回答问题：
1. 对于青少年，用生动有趣的语言，多举例子
2. 对于老年人，语速要慢，用词要清晰
3. 根据文化程度调整讲解深度
4. 保持友善、专业的态度
5. 回答要简洁明了，避免过于学术化的表达
6. 不要在回答中包含<think>等标记
7. 直接给出最终答案，不需要展示思考过程

请以自然的对话方式回答问题。"""
        
        return prompt
    
    async def generate_speech(self, text, voice="alloy", user_profile=None):
        """使用OpenAI TTS生成语音"""
        try:
            import hashlib
            
            # 根据用户语言选择声音
            if user_profile and user_profile.get('language') == 'English':
                voice = "nova"  # 英文使用nova声音
            else:
                # 根据用户性别选择中文声音
                if user_profile and user_profile.get('gender') == '女':
                    voice = "shimmer"  # 女性声音
                else:
                    voice = "alloy"    # 男性声音
            
            # 生成文本哈希用于缓存
            text_hash = hashlib.md5((text + voice).encode('utf-8')).hexdigest()
            
            # 检查缓存
            conn = sqlite3.connect('data/museum_guide.db')
            cursor = conn.cursor()
            cursor.execute('SELECT file_path FROM audio_cache WHERE text_hash = ?', (text_hash,))
            result = cursor.fetchone()
            
            if result and os.path.exists(result[0]):
                # 返回缓存的音频文件路径
                conn.close()
                return result[0]
            
            # 生成新的音频
            response = client.audio.speech.create(
                model="tts-1",  # 或使用 "tts-1-hd" 获得更高质量
                voice=voice,
                input=text,
                response_format="mp3"
            )
            
            # 保存音频文件
            audio_file_path = os.path.join(self.audio_cache_dir, f"{text_hash}.mp3")
            with open(audio_file_path, 'wb') as f:
                f.write(response.content)
            
            # 保存到缓存数据库
            cursor.execute('''
                INSERT OR REPLACE INTO audio_cache (text_hash, voice_model, file_path)
                VALUES (?, ?, ?)
            ''', (text_hash, voice, audio_file_path))
            conn.commit()
            conn.close()
            
            return audio_file_path
            
        except Exception as e:
            logger.error(f"生成语音失败: {e}")
            return None

# 初始化系统
guide_system = MuseumGuideSystem()

@app.route('/')
def index():
    return jsonify({
        "message": "四川博物院AI导览系统API",
        "version": "2.0.0",
        "status": "运行中",
        "features": ["OpenAI TTS", "智能缓存", "个性化语音"]
    })

@app.route('/api/generate_route', methods=['POST'])
def generate_route():
    """生成个性化参观路线"""
    try:
        user_profile = request.json
        
        # 记录访问日志
        log_visit(user_profile)
        
        # 生成路线
        route = guide_system.generate_personalized_route(user_profile)
        
        return jsonify({
            "success": True,
            "route": route,
            "message": "路线生成成功"
        })
        
    except Exception as e:
        logger.error(f"生成路线失败: {e}")
        return jsonify({
            "success": False,
            "message": "生成路线失败，请重试"
        }), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """处理用户对话"""
    try:
        data = request.json
        user_input = data.get('message', '')
        user_profile = data.get('user_profile', {})
        context = data.get('context', {})
        
        # 处理用户问题
        response = guide_system.process_user_question(user_input, user_profile, context)
        
        return jsonify({
            "success": True,
            "response": response
        })
        
    except Exception as e:
        logger.error(f"处理对话失败: {e}")
        return jsonify({
            "success": False,
            "response": "抱歉，我暂时无法回答您的问题，请您稍后再试。"
        }), 500

@app.route('/api/generate_speech', methods=['POST'])
def generate_speech_api():
    """生成语音API"""
    try:
        data = request.json
        text = data.get('text', '')
        user_profile = data.get('user_profile', {})
        
        if not text:
            return jsonify({
                "success": False,
                "message": "文本不能为空"
            }), 400
        
        # 异步生成语音（这里简化为同步）
        import asyncio
        audio_file_path = asyncio.run(guide_system.generate_speech(text, user_profile=user_profile))
        
        if audio_file_path and os.path.exists(audio_file_path):
            # 读取音频文件并转为base64
            with open(audio_file_path, 'rb') as f:
                audio_data = f.read()
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            return jsonify({
                "success": True,
                "audio_data": audio_base64,
                "content_type": "audio/mpeg",
                "message": "语音生成成功"
            })
        else:
            return jsonify({
                "success": False,
                "message": "语音生成失败"
            }), 500
            
    except Exception as e:
        logger.error(f"生成语音失败: {e}")
        return jsonify({
            "success": False,
            "message": "语音生成失败"
        }), 500

@app.route('/api/audio/<filename>')
def serve_audio(filename):
    """提供音频文件服务"""
    try:
        audio_file_path = os.path.join(guide_system.audio_cache_dir, filename)
        if os.path.exists(audio_file_path):
            return send_file(audio_file_path, mimetype='audio/mpeg')
        else:
            return jsonify({
                "success": False,
                "message": "音频文件不存在"
            }), 404
    except Exception as e:
        logger.error(f"提供音频文件失败: {e}")
        return jsonify({
            "success": False,
            "message": "音频文件服务失败"
        }), 500

@app.route('/api/navigation/<from_id>/<to_id>')
def get_navigation(from_id, to_id):
    """获取导航信息"""
    try:
        nav_info = guide_system.get_navigation_info(from_id, to_id)
        
        if nav_info:
            return jsonify({
                "success": True,
                "navigation": nav_info
            })
        else:
            return jsonify({
                "success": False,
                "message": "未找到导航路径"
            }), 404
            
    except Exception as e:
        logger.error(f"获取导航信息失败: {e}")
        return jsonify({
            "success": False,
            "message": "获取导航信息失败"
        }), 500

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """提交用户反馈"""
    try:
        data = request.json
        user_profile = data.get('user_profile', {})
        ratings = data.get('ratings', {})
        
        # 保存反馈到数据库
        conn = sqlite3.connect('data/museum_guide.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO feedback (user_profile, route_rating, content_rating, service_rating, interaction_rating)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            json.dumps(user_profile, ensure_ascii=False),
            ratings.get('route', 0),
            ratings.get('content', 0),
            ratings.get('service', 0),
            ratings.get('interaction', 0)
        ))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            "success": True,
            "message": "反馈提交成功"
        })
        
    except Exception as e:
        logger.error(f"提交反馈失败: {e}")
        return jsonify({
            "success": False,
            "message": "反馈提交失败"
        }), 500

@app.route('/api/artifacts')
def get_artifacts():
    """获取所有藏品信息"""
    try:
        return jsonify({
            "success": True,
            "artifacts": guide_system.artifacts_nodes
        })
    except Exception as e:
        logger.error(f"获取藏品信息失败: {e}")
        return jsonify({
            "success": False,
            "message": "获取藏品信息失败"
        }), 500

@app.route('/api/knowledge/<artifact_id>')
def get_knowledge(artifact_id):
    """获取特定藏品的知识库信息"""
    try:
        if artifact_id in guide_system.knowledge_base:
            return jsonify({
                "success": True,
                "knowledge": guide_system.knowledge_base[artifact_id]
            })
        else:
            return jsonify({
                "success": False,
                "message": "未找到该藏品的信息"
            }), 404
    except Exception as e:
        logger.error(f"获取知识库信息失败: {e}")
        return jsonify({
            "success": False,
            "message": "获取信息失败"
        }), 500

@app.route('/api/stats')
def get_stats():
    """获取系统统计信息"""
    try:
        conn = sqlite3.connect('data/museum_guide.db')
        cursor = conn.cursor()
        
        # 获取访问统计
        cursor.execute('SELECT COUNT(*) FROM visit_logs')
        total_visits = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM feedback')
        total_feedback = cursor.fetchone()[0]
        
        cursor.execute('SELECT AVG(route_rating), AVG(content_rating), AVG(service_rating), AVG(interaction_rating) FROM feedback')
        avg_ratings = cursor.fetchone()
        
        # 获取音频缓存统计
        cursor.execute('SELECT COUNT(*) FROM audio_cache')
        cached_audios = cursor.fetchone()[0]
        
        conn.close()
        
        return jsonify({
            "success": True,
            "stats": {
                "total_visits": total_visits,
                "total_feedback": total_feedback,
                "cached_audios": cached_audios,
                "average_ratings": {
                    "route": avg_ratings[0] or 0,
                    "content": avg_ratings[1] or 0,
                    "service": avg_ratings[2] or 0,
                    "interaction": avg_ratings[3] or 0
                }
            }
        })
        
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        return jsonify({
            "success": False,
            "message": "获取统计信息失败"
        }), 500

def log_visit(user_profile):
    """记录访问日志"""
    try:
        conn = sqlite3.connect('data/museum_guide.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO visit_logs (user_profile, tour_route, questions_asked)
            VALUES (?, ?, ?)
        ''', (
            json.dumps(user_profile, ensure_ascii=False),
            "",  # 稍后更新路线信息
            ""   # 稍后更新问题信息
        ))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        logger.error(f"记录访问日志失败: {e}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
