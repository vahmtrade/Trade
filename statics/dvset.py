from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities


break_time = 1
wait_time = 60

driver_options = Options()
driver_capabilities = DesiredCapabilities.FIREFOX

driver_ui = '{"placements":{"widget-overflow-fixed-list":[],"nav-bar":["back-button","forward-button","stop-reload-button","customizableui-special-spring1","urlbar-container","customizableui-special-spring2","save-to-pocket-button","fxa-toolbar-menu-button"],"toolbar-menubar":["menubar-items"],"TabsToolbar":["tabbrowser-tabs","new-tab-button","alltabs-button"],"PersonalToolbar":["import-button","personal-bookmarks"]},"seen":["save-to-pocket-button","developer-button"],"dirtyAreaCache":["nav-bar","PersonalToolbar","widget-overflow-fixed-list"],"currentVersion":17,"newElementCount":2}'

driver_options.set_preference("browser.download.folderList", 2)
driver_options.set_preference("browser.uiCustomization.state", driver_ui)
driver_options.set_preference("browser.download.autohideButton", False)
driver_options.set_preference("dom.security.https_only_mode", False)
driver_options.accept_insecure_certs = True
driver_options.headless = True

driver_capabilities["marionette"] = True


# driver_options.set_preference("browser.download.animateNotifications", False)
# driver_options.set_preference("browser.download.alwaysOpenPanel", False)
# driver_options.set_preference("browser.download.panel.shown", False)
# driver_options.set_preference("browser.download.manager.showWhenStarting", False)
# driver_options.set_preference("browser.download.manager.focusWhenStarting", False)
# driver_options.set_preference("browser.download.manager.closeWhenDone", True)
# driver_options.set_preference("browser.download.manager.showAlertOnComplete", False)
# driver_options.set_preference("browser.download.manager.useWindow", False)
# driver_options.set_preference("browser.preferences.instantApply", True)
# driver_options.set_preference("dom.webnotifications.enabled", False)
# driver_options.set_preference("dom.security.https_only_mode", False)
