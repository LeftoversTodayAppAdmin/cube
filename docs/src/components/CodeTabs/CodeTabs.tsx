import * as React from 'react';
import { useState, useEffect, useMemo, type FC } from 'react';
import { langs } from './dictionary';

import classnames from 'classnames/bind';
import * as classes from './CodeTabs.module.css';
const cn = classnames.bind(classes);

interface CustomEventMap {
  'codetabs.changed': CustomEvent<{ lang: string }>;
}

declare global {
  interface Window {
    addEventListener<K extends keyof CustomEventMap>(
      type: K,
      listener: (this: Document, ev: CustomEventMap[K]) => void
    ): void;
    removeEventListener<K extends keyof CustomEventMap>(
      type: K,
      listener: (this: Window, ev: CustomEventMap[K]) => any,
      options?: boolean | EventListenerOptions
    ): void;
    dispatchEvent<K extends keyof CustomEventMap>(ev: CustomEventMap[K]): void;
  }
}

const STORAGE_KEY = 'cube-docs.default-code-lang';

// If present, the tab with this language should go first
const PREFERRED_LANG = 'yaml';

export interface CodeTabsProps {
  children: Array<{
    props: {
      'data-language': string;
      children: any;
    };
  }>;
}

export const CodeTabs: FC<CodeTabsProps> = ({ children }) => {
  const [selectedTab, setSelectedTab] = useState(PREFERRED_LANG);
  const tabs = useMemo(
    () => {
      let tabs = children.reduce<Record<string, number>>((dict, tab, i) => {
        const result = {
          ...dict,
        };
        if (result[tab.props['data-language']] === undefined) {
          result[tab.props['data-language']] = i;
        }
        return result;
      }, {});
      
      // Place the tab with the prefferred lamguage on the first position
      let tabWithPreferredLangKey = Object.keys(tabs).find(key => key === PREFERRED_LANG);
      if (tabWithPreferredLangKey !== undefined) {
        let tabWithPreferredLangValue = tabs[tabWithPreferredLangKey];
        delete tabs[tabWithPreferredLangKey];
        tabs = {
          [tabWithPreferredLangKey]: tabWithPreferredLangValue,
          ...tabs
        };
      }

      return tabs;
    },
    children
  );

  useEffect(() => {
    const defaultLang = localStorage.getItem(STORAGE_KEY);

    if (defaultLang) {
      if (tabs[defaultLang] !== undefined) {
        setSelectedTab(defaultLang);
      }
    }

    const syncHanlder = (e: CustomEvent<{ lang: string }>) => {
      const lang = e.detail.lang;
      if (tabs[lang] !== undefined) {
        setSelectedTab(lang);
      }
    };

    const storageHandler = (e: StorageEvent) => {
      if (e.key === STORAGE_KEY) {
        const lang = e.newValue;
        if (lang && tabs[lang] !== undefined) {
          setSelectedTab(lang);
        }
      }
    };

    window.addEventListener('storage', storageHandler);
    window.addEventListener('codetabs.changed', syncHanlder);

    return () => {
      window.removeEventListener('storage', storageHandler);
      window.removeEventListener('codetabs.changed', syncHanlder);
    };
  }, []);

  return (
    <div className={classes.CodeBlock}>
      <div className={classes.CodeBlocks__tabs}>
        {Object
          .entries(tabs)
          .map(tab => children.find(child => child.props['data-language'] === tab[0]))
          .filter((tab) => tab !== undefined && !!tab.props['data-language'])
          .map((tab, i) => {
            if (tab === undefined) return;
            let lang = tab.props['data-language'];
            if (lang === 'js') {
              lang = 'javascript';
            }
            return (
              <div
                key={i}
                className={cn('CodeBlocks__tab', {
                  [classes.SelectedTab]: lang === selectedTab,
                })}
                onClick={() => {
                  if (
                    lang !== selectedTab &&
                    (lang === 'javascript' || lang === 'yaml')
                  ) {
                    localStorage.setItem(STORAGE_KEY, lang);
                    window.dispatchEvent(
                      new CustomEvent('codetabs.changed', {
                        detail: {
                          lang,
                        },
                      })
                    );
                  }
                  setSelectedTab(lang);
                }}
              >
                {langs[lang] || lang}
              </div>
            );
          })}
      </div>

      {children && children.find(child => child.props['data-language'] === selectedTab)?.props.children}
    </div>
  );
};
