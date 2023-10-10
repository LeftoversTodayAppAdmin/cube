import type { CommanderStatic } from 'commander';
import fs from 'fs-extra';
import rp from 'request-promise';

import { displayError, event } from '../utils';

const wrapInQuotes = (values: string[]) => {
  if (values.length === 0) {
    return '\'\'';
  }

  return values
    .map(m => `'${m}'`)
    .join('\n  | ');
};

const stripSpaces = (str: string): string => str.replace(/\s/g, '');

const selectName = (o) => o.name;

const createCubeProps = (cube: any) => {
  const cubeName = stripSpaces(cube.name);
  const dimensions = cube.dimensions.map(selectName);
  const measures = cube.measures.map(selectName);
  const timeDimensions = cube.dimensions.filter((d) => d.type === 'time').map(selectName);
  const segments = cube.segments.map(selectName);

  return `export type ${cubeName}Measure = ${wrapInQuotes(measures)};
export type ${cubeName}Dimension = ${wrapInQuotes(dimensions)};
export type ${cubeName}TimeDimension = ${wrapInQuotes(timeDimensions)};
export type ${cubeName}Segment = ${wrapInQuotes(segments)};
`;
};

const createTypeDefinitions = (cubes: any[]) => {
  const cubeNames = cubes.map(cube => cube.name);
  const topLevelTypes = [
    'Measure',
    'Dimension',
    'TimeDimension',
    'Segment',
  ];

  return `// DO NOT MANUALLY EDIT - THIS FILE IS AUTOMATICALLY GENERATED

export type Cubes = ${cubeNames.map(c => `'${c}'`).join('\n  | ')};

${cubes.map(createCubeProps).join('\n')}
${topLevelTypes.map((topLevelType) => `
export type Introspected${topLevelType}Name = ${cubeNames.map(c => `${c}${topLevelType}`).join('\n  | ')};
`).join('\n')}`;
};

const generateQueryTypes = async (apiUrl, { token }) => {
  if (!apiUrl) {
    await displayError('Please specify the Cube API URL');
  }
  if (!token) {
    await displayError('Type generation requires a token. Please generate one and try again');
  }

  let meta = {
    cubes: [],
  };

  try {
    meta = await rp({
      headers: {
        authorization: token,
      },
      url: `${apiUrl}/meta`,
      json: true,
    });
  } catch (e: any) {
    await displayError(e.error.error);
  }

  if (meta.cubes.length === 0) {
    await displayError('No cubes were found. Please create a cube and try again');
  }

  const typeDefs = createTypeDefinitions(meta.cubes);
  const pathToCubeDX = require.resolve('@cubejs-client/dx', {
    paths: [process.cwd()],
  });
  const pathToDefinitionFile = pathToCubeDX.replace('/src/index.js', '/generated.d.ts');

  try {
    await fs.writeFile(pathToDefinitionFile, typeDefs);
    await event({
      event: 'cli:typegen:success',
      area: 'cli:typegen'
    });
    console.log('Types successfully generated!');
  } catch (e) {
    await displayError('Could not persist types to disk. Please file a GitHub issue reporting this', {
      area: 'cli:typegen',
    });
  }
};

export function configureTypegenCommand(program: CommanderStatic): void {
  program
    .command('typegen <apiUrl>')
    .description('Generate types from your Cube API for your frontend project')
    .option('--token <token>', 'A valid JWT for your Cube project')
    .action(
      (apiUrl, options) => generateQueryTypes(apiUrl, options)
        .catch(e => displayError(e.stack || e))
    )
    .on('--help', () => {
      console.log('');
      console.log('Example:');
      console.log('');
      console.log('  $ cubejs typegen http://localhost:4000/cubejs-api/v1 --token eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJkZXBsb3ltZW50SWQiOiIxIiwidXJsIjoiaHR0cHM6Ly9leGFtcGxlcy5jdWJlY2xvdWQuZGV2IiwiaWF0IjoxNTE2MjM5MDIyfQ.La3MiuqfGigfzADl1wpxZ7jlb6dY60caezgqIOoHt-c');
    });
}
