module.exports = {
  extends: [
    'react-app',
    'react-app/jest'
  ],
  rules: {
    // Allow unused variables that start with underscore
    'no-unused-vars': ['error', { 
      'argsIgnorePattern': '^_',
      'varsIgnorePattern': '^_'
    }],
    // Don't treat warnings as errors in production builds
    'react-hooks/exhaustive-deps': 'warn'
  },
  env: {
    browser: true,
    es6: true,
    node: true
  }
};
