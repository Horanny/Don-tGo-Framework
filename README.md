This repository contains the framework for ***Deciphering Explicit and Implicit Features for Reliable, Interpretable, and Actionable User Churn Prediction in Online Video Games.***

It consists of two components:

- **Interface**: Contains frontend code.
- **Backend**: Contains backend code.

## Interface Setup

This framework is based on Node.js and npm.

- To install Node.js, please refer to https://nodejs.org/en.
- To install npm, please refer to https://www.npmjs.com/.
- We recommend using nvm (Node Version Manager) for easy Node.js version management: https://github.com/nvm-sh/nvm.

Required versions:

```jsx
node version: v16.15.0
npm version: v9.6.0
```

### Installation

Once the environment is set up, install the required node modules by running:

```jsx
npm install
```

### Backend IP Configuration

In `interface/config/index.js`, replace `YOUR_BACKEND_IP` with the actual backend IP:

```jsx
dev: {
    env: require("./dev.env"),
    port: 8050,
    assetsSubDirectory: "static",
    assetsPublicPath: "/",
    proxyTable: {
      '/cf': {
        target: `http://YOUR_BACKEND_IP:${proxyPort}`,
        changeOrigin: true,
        pathRewrite: {
          '^/cf': '/cf'
        }
      },
    },
    cssSourceMap: false
  }
```

### Launch the Interface

To launch the **Don’tGo** interface, run the following command:

```jsx
npm run start
```

## Backend Setup

To run the backend, execute:

```jsx
python app.py
```