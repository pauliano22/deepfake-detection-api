"""
API Testing Module
Consolidated testing for the voice detection API.
"""

import requests
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

class ApiTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.test_results = []
    
    def test_health_endpoint(self) -> bool:
        """Test API health endpoint"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                model_loaded = data.get('model_loaded', False)
                
                print(f"✅ Health endpoint: {response.status_code}")
                print(f"   Model loaded: {'Yes' if model_loaded else 'No'}")
                
                return model_loaded
            else:
                print(f"❌ Health endpoint failed: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Health endpoint error: {e}")
            return False
    
    def test_single_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Test detection on a single file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            return None
        
        try:
            with open(file_path, 'rb') as f:
                files = {'audio': f}
                response = requests.post(f"{self.base_url}/detect", files=files, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"🎵 Test: {file_path.name}")
                print(f"   Detected as: {'AI Generated' if result['is_ai_generated'] else 'Human Voice'}")
                print(f"   Confidence: {result['confidence']:.3f}")
                print(f"   AI Probability: {result['ai_probability']:.3f}")
                print(f"   Human Probability: {result['human_probability']:.3f}")
                
                return result
            else:
                print(f"❌ Detection failed for {file_path.name}: {response.status_code}")
                print(f"   Error: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Error testing {file_path.name}: {e}")
            return None
    
    def test_batch_files(self, file_list: List[str]) -> List[Dict[str, Any]]:
        """Test detection on multiple files"""
        results = []
        
        print(f"🧪 Testing {len(file_list)} files...")
        
        for file_path in file_list:
            result = self.test_single_file(file_path)
            if result:
                results.append({
                    'file': file_path,
                    'result': result
                })
            print()  # Empty line for readability
        
        return results
    
    def test_data_directory(self) -> Dict[str, List[Dict[str, Any]]]:
        """Test all files in the data directory"""
        ai_dir = Path("data/raw/ai")
        human_dir = Path("data/raw/human")
        
        results = {
            'ai_samples': [],
            'human_samples': []
        }
        
        # Test AI samples
        if ai_dir.exists():
            ai_files = list(ai_dir.glob("*.wav"))
            if ai_files:
                print("🤖 Testing AI samples:")
                print("=" * 30)
                
                for file_path in ai_files[:5]:  # Test first 5 files
                    result = self.test_single_file(file_path)
                    if result:
                        results['ai_samples'].append({
                            'file': str(file_path),
                            'result': result,
                            'correct': result['is_ai_generated']  # Should be True
                        })
        
        # Test human samples
        if human_dir.exists():
            human_files = list(human_dir.glob("*.wav"))
            if human_files:
                print("\n👤 Testing human samples:")
                print("=" * 30)
                
                for file_path in human_files[:5]:  # Test first 5 files
                    result = self.test_single_file(file_path)
                    if result:
                        results['human_samples'].append({
                            'file': str(file_path),
                            'result': result,
                            'correct': not result['is_ai_generated']  # Should be False
                        })
        
        return results
    
    def test_error_cases(self):
        """Test API error handling"""
        print("\n🚨 Testing error cases:")
        print("=" * 30)
        
        # Test invalid file format
        try:
            with open("README.md", 'rb') as f:
                files = {'audio': f}
                response = requests.post(f"{self.base_url}/detect", files=files, timeout=10)
            
            if response.status_code == 400:
                print("✅ Invalid file format correctly rejected")
            else:
                print(f"❌ Invalid file format test failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error testing invalid format: {e}")
        
        # Test missing file
        try:
            response = requests.post(f"{self.base_url}/detect", timeout=10)
            
            if response.status_code == 422:
                print("✅ Missing file correctly rejected")
            else:
                print(f"❌ Missing file test failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error testing missing file: {e}")
    
    def benchmark_performance(self, file_path: str, num_requests: int = 10):
        """Benchmark API performance"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            print(f"❌ Benchmark file not found: {file_path}")
            return
        
        print(f"\n⏱️ Benchmarking performance ({num_requests} requests):")
        print("=" * 50)
        
        times = []
        
        for i in range(num_requests):
            start_time = time.time()
            
            try:
                with open(file_path, 'rb') as f:
                    files = {'audio': f}
                    response = requests.post(f"{self.base_url}/detect", files=files, timeout=30)
                
                end_time = time.time()
                request_time = end_time - start_time
                times.append(request_time)
                
                if response.status_code == 200:
                    print(f"  Request {i+1:2d}: {request_time:.3f}s ✅")
                else:
                    print(f"  Request {i+1:2d}: {request_time:.3f}s ❌ ({response.status_code})")
                    
            except Exception as e:
                end_time = time.time()
                request_time = end_time - start_time
                print(f"  Request {i+1:2d}: {request_time:.3f}s ❌ ({e})")
        
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            print(f"\n📊 Performance Summary:")
            print(f"   Average: {avg_time:.3f}s")
            print(f"   Min:     {min_time:.3f}s")
            print(f"   Max:     {max_time:.3f}s")
            print(f"   RPS:     {1/avg_time:.1f} requests/second")
    
    def run_comprehensive_tests(self):
        """Run all tests"""
        print("🧪 Running Comprehensive API Tests")
        print("=" * 50)
        
        # Test health endpoint
        print("1. Testing health endpoint...")
        health_ok = self.test_health_endpoint()
        
        if not health_ok:
            print("❌ API not ready for testing (model not loaded)")
            return False
        
        # Test data directory
        print("\n2. Testing with training data...")
        data_results = self.test_data_directory()
        
        # Calculate accuracy
        ai_correct = sum(1 for r in data_results['ai_samples'] if r['correct'])
        human_correct = sum(1 for r in data_results['human_samples'] if r['correct'])
        total_ai = len(data_results['ai_samples'])
        total_human = len(data_results['human_samples'])
        
        if total_ai > 0 and total_human > 0:
            ai_accuracy = ai_correct / total_ai
            human_accuracy = human_correct / total_human
            overall_accuracy = (ai_correct + human_correct) / (total_ai + total_human)
            
            print(f"\n📊 Accuracy Results:")
            print(f"   AI Detection:    {ai_accuracy:.1%} ({ai_correct}/{total_ai})")
            print(f"   Human Detection: {human_accuracy:.1%} ({human_correct}/{total_human})")
            print(f"   Overall:         {overall_accuracy:.1%}")
        
        # Test error cases
        print("\n3. Testing error handling...")
        self.test_error_cases()
        
        # Benchmark performance (if we have test files)
        if total_ai > 0:
            test_file = data_results['ai_samples'][0]['file']
            print(f"\n4. Benchmarking performance...")
            self.benchmark_performance(test_file, 5)
        
        print("\n✅ Comprehensive testing completed!")
        return True

def main():
    """Run API tests from command line"""
    import sys
    
    if len(sys.argv) > 1:
        endpoint = sys.argv[1]
    else:
        endpoint = "http://localhost:8000"
    
    tester = ApiTester(endpoint)
    tester.run_comprehensive_tests()

if __name__ == "__main__":
    main()