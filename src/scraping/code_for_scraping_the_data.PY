import requests
from bs4 import BeautifulSoup
import pandas as pd
import csv
import time
import random
import threading
from queue import Queue
import concurrent.futures
import logging
from fake_useragent import UserAgent
import urllib3
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# تعطيل تحذيرات SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# إعداد التسجيل
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GuardianNewsScraper:
    def __init__(self):
        self.output_file = r"D:\Level 3\AL\articles_data.csv"
        self.articles_queue = Queue()
        self.articles_count = 0
        self.target_count = 100000
        self.ua = UserAgent()
        
        # مصدر واحد فقط - The Guardian
        self.news_sources = [
            {'name': 'The Guardian', 'url': 'https://www.theguardian.com/international', 'category': 'news'},
        ]
        
        self.setup_session()
        self.initialize_csv()
    
    def setup_session(self):
        """إعداد session مع retry strategy"""
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=100, pool_maxsize=100)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def initialize_csv(self):
        """تهيئة ملف CSV"""
        try:
            with open(self.output_file, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['id', 'source', 'category', 'title', 'link', 'summary', 'published_date', 'scraped_at'])
            logger.info("تم تهيئة ملف CSV")
        except Exception as e:
            logger.error(f"خطأ في تهيئة CSV: {e}")
    
    def get_headers(self):
        """إنشاء headers عشوائية"""
        return {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
    
    def scrape_guardian(self, source):
        """سحب من The Guardian"""
        try:
            response = self.session.get(source['url'], headers=self.get_headers(), timeout=10, verify=False)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            articles = []
            
            # البحث عن المقالات في الأقسام المختلفة
            selectors = [
                'a[data-link-name="article"]',
                '.fc-item__title a',
                '.js-headline-text',
                '.u-faux-block-link__overlay',
                '.dcr-1bfp2m6'
            ]
            
            for selector in selectors:
                for item in soup.select(selector)[:100]:  # زيادة العدد لسحب المزيد
                    title = item.get_text(strip=True)
                    link = item.get('href', '')
                    
                    if title and len(title) > 20 and link:
                        # التأكد من أن الرابط كامل
                        full_link = link
                        if link.startswith('//'):
                            full_link = 'https:' + link
                        elif link.startswith('/'):
                            full_link = 'https://www.theguardian.com' + link
                        
                        articles.append({
                            'source': source['name'],
                            'category': source['category'],
                            'title': title[:500],
                            'link': full_link,
                            'summary': '',
                            'published_date': self.extract_guardian_date(item),
                            'scraped_at': pd.Timestamp.now()
                        })
            
            # إذا لم نجد مقالات كافية، نستخدم طريقة بديلة
            if len(articles) < 50:
                for item in soup.find_all('a', href=True)[:200]:
                    title = item.get_text(strip=True)
                    link = item.get('href', '')
                    
                    if (title and len(title) > 30 and 
                        any(keyword in link for keyword in ['/202', '/article', '/news'])):
                        
                        full_link = link
                        if link.startswith('//'):
                            full_link = 'https:' + link
                        elif link.startswith('/'):
                            full_link = 'https://www.theguardian.com' + link
                        
                        # تجنب الروابط المكررة
                        if not any(art['link'] == full_link for art in articles):
                            articles.append({
                                'source': source['name'],
                                'category': source['category'],
                                'title': title[:500],
                                'link': full_link,
                                'summary': '',
                                'published_date': '',
                                'scraped_at': pd.Timestamp.now()
                            })
            
            return articles[:100]  # تحديد العدد الأقصى
        except Exception as e:
            logger.error(f"خطأ في The Guardian: {e}")
            return []
    
    def extract_guardian_date(self, element):
        """استخراج تاريخ النشر من العنصر"""
        try:
            # البحث عن العنصر الأب الذي قد يحتوي على التاريخ
            parent = element.find_parent()
            if parent:
                time_element = parent.find('time')
                if time_element and time_element.get('datetime'):
                    return time_element.get('datetime')
        except:
            pass
        return ''
    
    def scrape_guardian_sections(self):
        """سحب المقالات من أقسام مختلفة في The Guardian"""
        sections = [
            'https://www.theguardian.com/international',
            'https://www.theguardian.com/uk',
            'https://www.theguardian.com/world',
            'https://www.theguardian.com/politics',
            'https://www.theguardian.com/sport',
            'https://www.theguardian.com/football',
            'https://www.theguardian.com/technology',
            'https://www.theguardian.com/business',
            'https://www.theguardian.com/culture',
            'https://www.theguardian.com/lifeandstyle'
        ]
        
        all_articles = []
        source = {'name': 'The Guardian', 'category': 'news'}
        
        for section_url in sections:
            try:
                logger.info(f"جاري سحب البيانات من قسم: {section_url}")
                response = self.session.get(section_url, headers=self.get_headers(), timeout=10, verify=False)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # سحب المقالات من هذا القسم
                articles = self.scrape_guardian_from_soup(soup, source)
                all_articles.extend(articles)
                
                # انتظار عشوائي بين الطلبات
                time.sleep(random.uniform(1, 3))
                
            except Exception as e:
                logger.error(f"خطأ في سحب القسم {section_url}: {e}")
                continue
        
        return all_articles
    
    def scrape_guardian_from_soup(self, soup, source):
        """سحب المقالات من صفحة BeautifulSoup"""
        articles = []
        
        # البحث في عناصر المقالات المختلفة
        article_selectors = [
            'a[data-link-name="article"]',
            '.fc-item__title a',
            '.js-headline-text',
            '.dcr-1bfp2m6',  # Selector جديد
            '.dcr-1d8k5y1',  # Selector إضافي
        ]
        
        for selector in article_selectors:
            for item in soup.select(selector):
                try:
                    title = item.get_text(strip=True)
                    link = item.get('href', '')
                    
                    if title and len(title) > 20 and link:
                        full_link = self.normalize_guardian_link(link)
                        
                        # تجنب التكرار
                        if not any(art['link'] == full_link for art in articles):
                            articles.append({
                                'source': source['name'],
                                'category': self.determine_category(full_link),
                                'title': title[:500],
                                'link': full_link,
                                'summary': '',
                                'published_date': self.extract_date_from_element(item),
                                'scraped_at': pd.Timestamp.now()
                            })
                except Exception as e:
                    continue
        
        return articles
    
    def normalize_guardian_link(self, link):
        """تطبيع رابط The Guardian"""
        if link.startswith('//'):
            return 'https:' + link
        elif link.startswith('/'):
            return 'https://www.theguardian.com' + link
        else:
            return link
    
    def determine_category(self, link):
        """تحديد الفئة بناءً على الرابط"""
        if '/sport/' in link or '/football/' in link:
            return 'sports'
        elif '/politics/' in link:
            return 'politics'
        elif '/technology/' in link:
            return 'technology'
        elif '/business/' in link:
            return 'business'
        elif '/culture/' in link:
            return 'culture'
        else:
            return 'news'
    
    def extract_date_from_element(self, element):
        """استخراج التاريخ من العنصر"""
        try:
            # البحث في العناصر المحيطة
            parent = element.find_parent()
            for _ in range(3):  # البحث في 3 مستويات للأعلى
                if parent:
                    time_element = parent.find('time')
                    if time_element and time_element.get('datetime'):
                        return time_element.get('datetime')
                    parent = parent.find_parent()
        except:
            pass
        return ''
    
    def scrape_source(self, source):
        """سحب المقالات من مصدر معين"""
        logger.info(f"جاري سحب البيانات من: {source['name']}")
        
        if 'guardian' in source['url'].lower():
            articles = self.scrape_guardian(source)
        else:
            articles = []
        
        return articles
    
    def save_articles_batch(self, articles_batch):
        """حفظ دفعة من المقالات"""
        try:
            if not articles_batch:
                return
            
            # إضافة ID فريد لكل مقالة
            for i, article in enumerate(articles_batch):
                article['id'] = f"{article['source']}_{self.articles_count + i}_{int(time.time())}"
            
            # حفظ في CSV
            df_batch = pd.DataFrame(articles_batch)
            df_batch.to_csv(self.output_file, mode='a', header=False, index=False, encoding='utf-8')
            
            self.articles_count += len(articles_batch)
            logger.info(f"تم حفظ {len(articles_batch)} مقالة - الإجمالي: {self.articles_count}")
            
        except Exception as e:
            logger.error(f"خطأ في حفظ الدفعة: {e}")
    
    def run_scraping(self):
        """تشغيل عملية السحب الرئيسية"""
        logger.info("بدء عملية سحب 100,000 مقالة من The Guardian...")
        
        iteration = 0
        while self.articles_count < self.target_count:
            iteration += 1
            logger.info(f"== الدورة {iteration} - الإجمالي الحالي: {self.articles_count} ==")
            
            all_articles = []
            
            # سحب من الصفحة الرئيسية
            for source in self.news_sources:
                articles = self.scrape_source(source)
                all_articles.extend(articles)
            
            # سحب من الأقسام المختلفة
            section_articles = self.scrape_guardian_sections()
            all_articles.extend(section_articles)
            
            # إزالة التكرار
            unique_articles = []
            seen_links = set()
            for article in all_articles:
                if article['link'] not in seen_links:
                    seen_links.add(article['link'])
                    unique_articles.append(article)
            
            # حفظ الدفعة
            if unique_articles:
                self.save_articles_batch(unique_articles)
            
            # إظهار التقدم
            progress = (self.articles_count / self.target_count) * 100
            logger.info(f"التقدم: {progress:.1f}% - {self.articles_count}/{self.target_count}")
            
            # إذا لم نحصل على بيانات جديدة، ننتظر أكثر
            if not unique_articles:
                logger.warning("لم يتم سحب أي مقالات في هذه الدورة، زيادة وقت الانتظار")
                time.sleep(10)
            else:
                # وقت انتظار عشوائي لتجنب الحظر
                wait_time = random.uniform(5, 15)
                logger.info(f"انتظار {wait_time:.1f} ثانية قبل الدورة التالية")
                time.sleep(wait_time)
            
            # إذا استغرقت أكثر من 50 دورة بدون تقدم كافٍ، نوقف
            if iteration > 50 and self.articles_count < 1000:
                logger.error("عملية السحب بطيئة جداً، تحقق من الاتصال بالإنترنت أو هيكل الموقع")
                break
        
        logger.info(f"اكتملت عملية السحب! تم جمع {self.articles_count} مقالة من The Guardian")

def main():
    """الدالة الرئيسية"""
    try:
        scraper = GuardianNewsScraper()
        scraper.run_scraping()
        
        # عرض إحصائية نهائية
        if scraper.articles_count > 0:
            try:
                df = pd.read_csv(scraper.output_file)
                print(f"\n{'='*50}")
                print("الإحصائية النهائية:")
                print(f"إجمالي المقالات: {len(df)}")
                print(f"التوزيع حسب الفئة:")
                print(df['category'].value_counts())
                print(f"تم الحفظ في: {scraper.output_file}")
                print(f"{'='*50}")
            except Exception as e:
                print(f"خطأ في قراءة الملف للإحصائية: {e}")
        
    except Exception as e:
        logger.error(f"خطأ رئيسي: {e}")

if __name__ == "__main__":
    main()