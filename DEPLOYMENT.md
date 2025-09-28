# 🚀 F1 Prediction System - Railway Deployment Guide

## 📋 Prerequisites

- GitHub account with your F1 repository
- Railway account (free)
- Your code pushed to GitHub

## 🚀 Step-by-Step Railway Deployment

### Step 1: Create Railway Account
1. Go to [railway.app](https://railway.app)
2. Click **"Sign Up"**
3. Choose **"Sign up with GitHub"**
4. Authorize Railway to access your GitHub account

### Step 2: Deploy Your Project
1. In Railway dashboard, click **"New Project"**
2. Select **"Deploy from GitHub repo"**
3. Find and select your `f1-strategy-sim` repository
4. Click **"Deploy Now"**

### Step 3: Configure Environment Variables
In your Railway project dashboard:

1. Go to **"Variables"** tab
2. Add these environment variables:

```
PYTHONPATH=/app/src
FASTAPI_ENV=production
PORT=8000
```

### Step 4: Configure Build Settings
Railway will auto-detect Python, but you can verify:

1. Go to **"Settings"** tab
2. **Build Command**: `pip install -r web/backend/requirements.txt && pip install -e .`
3. **Start Command**: `uvicorn web.backend.main:app --host 0.0.0.0 --port $PORT`

### Step 5: Access Your Application
- **Your API URL**: `https://your-app-name.up.railway.app`
- **API Documentation**: `https://your-app-name.up.railway.app/docs`
- **Health Check**: `https://your-app-name.up.railway.app/health`

## 🔧 Railway Configuration Files

### railway.toml
```toml
[build]
builder = "nixpacks"

[deploy]
startCommand = "uvicorn web.backend.main:app --host 0.0.0.0 --port $PORT"
healthcheckPath = "/health"
healthcheckTimeout = 100
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 10

[env]
PYTHONPATH = "/app/src"
FASTAPI_ENV = "production"
```

### Procfile
```
web: uvicorn web.backend.main:app --host 0.0.0.0 --port $PORT
```

## 📊 Railway Free Tier Limits

- **$5 credit per month** (usually enough for small apps)
- **500 hours of usage**
- **1GB RAM**
- **1GB storage**
- **Custom domains**
- **Automatic deployments**

## 🎯 Testing Your Deployment

### 1. Health Check
```bash
curl https://your-app.up.railway.app/health
```

### 2. API Documentation
Visit: `https://your-app.up.railway.app/docs`

### 3. Test Prediction
```bash
curl -X POST "https://your-app.up.railway.app/predict" \
  -H "Content-Type: application/json" \
  -d '{"season": 2024, "race_round": 1, "session": "R"}'
```

## 🔄 Automatic Deployments

Railway automatically deploys when you push to your main branch:

```bash
# Make changes to your code
git add .
git commit -m "Update prediction model"
git push origin main

# Railway will automatically redeploy!
```

## 📱 Frontend Deployment (Optional)

For the React frontend, you can:

### Option 1: Deploy to Vercel (Free)
1. Go to [vercel.com](https://vercel.com)
2. Import your GitHub repository
3. Set **Root Directory** to `web/frontend`
4. Deploy!

### Option 2: Deploy to Netlify (Free)
1. Go to [netlify.com](https://netlify.com)
2. Connect your GitHub repository
3. Set **Base Directory** to `web/frontend`
4. Set **Build Command** to `npm run build`
5. Set **Publish Directory** to `build`

## 🐛 Troubleshooting

### Common Issues:

#### 1. Build Fails
- Check that all dependencies are in `web/backend/requirements.txt`
- Ensure Python version is compatible (3.11+)

#### 2. App Crashes on Startup
- Check environment variables are set correctly
- Verify PORT environment variable is used

#### 3. Health Check Fails
- Ensure `/health` endpoint returns 200 status
- Check logs in Railway dashboard

#### 4. Out of Memory
- Railway free tier has 1GB RAM limit
- Consider optimizing model loading
- Use smaller models for free tier

### View Logs:
```bash
# In Railway dashboard, go to "Deployments" tab
# Click on latest deployment to view logs
```

## 💰 Cost Management

### Free Tier Usage:
- Monitor usage in Railway dashboard
- $5 credit usually lasts the month for small apps
- Consider upgrading to paid plan if needed

### Optimize for Free Tier:
- Use smaller models
- Implement caching
- Optimize startup time
- Use environment variables for configuration

## 🎉 Success!

Once deployed, your F1 Prediction System will be available at:
- **API**: `https://your-app.up.railway.app`
- **Docs**: `https://your-app.up.railway.app/docs`
- **Health**: `https://your-app.up.railway.app/health`

## 🔗 Useful Links

- [Railway Documentation](https://docs.railway.app/)
- [Railway Pricing](https://railway.app/pricing)
- [FastAPI Deployment Guide](https://fastapi.tiangolo.com/deployment/)
- [Your GitHub Repository](https://github.com/WaqasAhmed27/f1-strategy-sim)

---

**Happy Predicting! 🏎️**
