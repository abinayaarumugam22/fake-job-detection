async function analyzeJob() {
            const jobText = document.getElementById('jobText').value.trim();
            
            if (!jobText) {
                alert('Please paste a job description first!');
                return;
            }

            // Show loading, hide result
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').style.display = 'none';

            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        job_description: jobText
                    })
                });

                const data = await response.json();

                if (data.error) {
                    alert(data.message);
                    document.getElementById('loading').style.display = 'none';
                    return;
                }

                // Display results
                displayResults(data);

            } catch (error) {
                console.error('Error:', error);
                alert('Failed to analyze job posting. Please try again.');
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        }

        function displayResults(data) {
            const resultDiv = document.getElementById('result');
            
            // Set class based on prediction
            resultDiv.className = 'result ' + (data.is_fake ? 'fake' : 'real');
            
            // Set title
            document.getElementById('resultTitle').textContent = 
                data.is_fake ? '🚨 WARNING: Potential Fake Job!' : '✅ Job Appears Legitimate';
            
            // Set details
            document.getElementById('prediction').textContent = data.prediction;
            document.getElementById('confidence').textContent = data.confidence + '%';
            document.getElementById('riskLevel').textContent = data.risk_level;
            document.getElementById('warning').textContent = data.warning;
            
            // Show result
            resultDiv.style.display = 'block';
            resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }

        function clearAll() {
            document.getElementById('jobText').value = '';
            document.getElementById('result').style.display = 'none';
            document.getElementById('loading').style.display = 'none';
        }

        function loadExample() {
            const exampleJob = `Marketing Intern

We're Food52, and we've created a groundbreaking and award-winning cooking site. We support and connect home cooks at every level through our recipes, hotline, contests, and online shop.

Job Overview:
We are looking for a driven, organized, and curious intern to join our team and support our growing business.

Responsibilities:
- Assist in content creation and social media management
- Support marketing campaigns and analytics
- Coordinate with team members on various projects
- Help with market research and competitor analysis

Requirements:
- Currently enrolled in Marketing, Communications, or related field
- Strong written and verbal communication skills
- Familiarity with social media platforms
- Ability to work 20 hours per week

Benefits:
- Internship stipend provided
- Flexible work schedule
- Mentorship from experienced professionals
- Potential for full-time position`;

            document.getElementById('jobText').value = exampleJob;
        }

        // Allow Enter key to submit
        document.getElementById('jobText').addEventListener('keydown', function(e) {
            if (e.ctrlKey && e.key === 'Enter') {
                analyzeJob();
            }
        });