const { OpenAI } = require('openai');
const { clipboard } = require('electron');


class ExplainInfo {
  constructor() {
    this.openai = new OpenAI({ baseURL: process.env.OPENAI_BASE_URL, apiKey: process.env.OPENAI_API_KEY, dangerouslyAllowBrowser: true });
    this.prompt = "";
    this.answer = "";
  }

  async explainThis() {
    const text = clipboard.readText()
    this.prompt = text;
    await this.callModel();
    this.displayExplanation();

  }

  async callModel() {
    // call open ai model to explain shit and return response
    const completion = await this.openai.chat.completions.create({
      model: "anthropic.claude-3.5-sonnet.v2",
      litellm_params: {
        modify_params: true
      },
      messages: [
        { role: "developer", content: "The user is a student, who is struggling with understanding the text they've given. explain the text given by the user in an easily understandable way" },
        {
          role: "user",
          content: this.prompt,
        },
      ],
    });
    const escapeHtml = (str) => str.replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/&/g, "&amp;");
    this.answer = escapeHtml(completion.choices[0].message.content).trim()
      .split(/\n\s*\n/) // Split on double newlines for paragraphs
      .map(para => `<p>${para.replace(/\n/g, "<br /> <br /> ")}</p>`) // Convert single newlines to <br>
      .join("\n")
  }

  displayExplanation() {
    const { BrowserWindow } = require('electron')
    const win = new BrowserWindow({
      height: 600,
      width: 800
    });
    // Enable @electron/remote for this webContents
    require("@electron/remote/main").enable(win.webContents);
    console.log(this.prompt, this.answer)
    win.loadURL(`file://${__dirname}/explain.html?prompt=${encodeURIComponent(this.prompt)}&answer=${this.answer}`);
  }
}


module.exports = { ExplainInfo }
