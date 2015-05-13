import numpy as np

# set random number seed
np.random.seed(0)

# load raw data
raw = dict()
for typ in ['stats', 'bowls']:
    raw[typ] = dict()
    for year in range(2002,2015):
        raw[typ][year] = dict()

        # direct to locations of stats and bowl results files
        if typ == 'stats':
            fname = 'Data/stats/cfb' + str(year) + 'stats.csv'
        if typ == 'bowls':
            fname = 'Data/bowls/bowl' + str(year) + 'lines.csv'
        
        with open(fname,'r') as fh:

            lines = fh.readlines()
            header = lines[0].split('\r')
            header = header[0].split('\n')
            header = header[0].split(',')

            # populate keys of dictionary
            for col in header:
                raw[typ][year][col] = list()

            # fill dictionary with values
            for line in lines[1:]:
                words = line.split(',')
                for i, word in enumerate(words):
                    try:
                        val = float(word)
                    except ValueError:
                        val = str(word)

                    raw[typ][year][header[i]].append(val)

# define alternate team names
alt_name = dict()
alt_name['UTEP'] = 'Texas-El Paso'
alt_name['U-A-B'] = 'Alabama-Birmingham'
alt_name['Troy'] = 'Troy State'

# delete games with unknown scores
for year in range(2002,2015):
    delete_ind = set()
    for i, val in enumerate(raw['stats'][year]['ScoreOff']):
        if val == ' ':
            delete_ind.add(i)
    if len(delete_ind) > 0:
        for col in raw['stats'][year].keys():
            new_list = list()
            for i, val in enumerate(raw['stats'][year][col]):
                if i not in delete_ind:
                    new_list.append(val)
            raw['stats'][year][col] = new_list

# create a set of bowl teams for each year
bowl_teams = dict()
for year in range(2002,2014):
    bowl_teams[year] = set()
    for team_status in ['Home Team', 'Visitor']:
        teams = raw['bowls'][year][team_status]
        for team in teams:
            bowl_teams[year].add(team)

# get list of game indexes for each team
team_games = dict()
opp_games = dict()
for year in range(2002,2014):
    team_games[year] = dict()
    for i, team in enumerate(raw['stats'][year]['TeamName']):
        if team not in team_games[year]:
            team_games[year][team] = [i]
        else:
            team_games[year][team].append(i)
    opp_games[year] = dict()
    for i, team in enumerate(raw['stats'][year]['Opponent']):
        if team not in opp_games[year]:
            opp_games[year][team] = [i]
        else:
            opp_games[year][team].append(i)

# define which entries not to record for stats
pass_attr = set(['TeamName', 'Opponent', 'Line', 'Site', 'Date'])

# form team data for each year
team_data = dict()
for year in range(2002,2014):
    team_data[year] = dict()
    for team in bowl_teams[year]:

        # check that team name is in the dictionary
        if team not in team_games[year]: 
            team = alt_name[team]
        if team not in team_games[year]:
            print "Error: failed to find team named ", team
            exit()
        
        team_data[year][team] = dict()

        # populate dictionary
        for typ in ['all','bowl']:
            team_data[year][team][typ] = dict()
            for stat in raw['stats'][year].keys():
                if stat in pass_attr:
                    continue
                else:
                    team_data[year][team][typ][stat] = 0
            team_data[year][team][typ]['Matches'] = 0
            team_data[year][team][typ]['Wins'] = 0
            team_data[year][team][typ]['Losses'] = 0
                
        # get summed attributes for each requested attribute
        for game_ind in team_games[year][team]:
            
            # check if opponent is bowl-bound
            bowl_bound_opp = raw['stats'][year]['Opponent'][game_ind] \
                    in bowl_teams[year]
            
            # add stats for each category from the game
            for stat in raw['stats'][year].keys():
                if stat in pass_attr:
                    continue
                else:
                    stat_val = raw['stats'][year][stat][game_ind]
                    stat_val = float(stat_val)

                    team_data[year][team]['all'][stat] += stat_val
                    if bowl_bound_opp:
                        team_data[year][team]['bowl'][stat] += stat_val

            # determine win or loss
            if raw['stats'][year]['ScoreOff'][game_ind] \
                    > raw['stats'][year]['ScoreDef'][game_ind]:
                result = 'Wins'
            else:
                result = 'Losses'

            # increment number of games, record result
            team_data[year][team]['all']['Matches'] += 1
            team_data[year][team]['all'][result] += 1
            if bowl_bound_opp:
                team_data[year][team]['bowl']['Matches'] += 1
                team_data[year][team]['bowl'][result] += 1

# make game-averaged stats and record winning %
for year in team_data:
    for team in team_data[year]:
        for typ in ['all', 'bowl']:

            # record winning percentage
            season_stats = team_data[year][team][typ]
            N = season_stats['Matches']
            if N == 0:
                season_stats['WinPct'] = 0.0
            else:
                season_stats['WinPct'] = float(season_stats['Wins']) / N
            
            # normalize stats by the number of games
            for stat in raw['stats'][year].keys():
                if stat in pass_attr:
                    continue
                else:
                    if N != 0:
                        season_stats[stat] /= N

# Add derived features
for year in team_data:
    for team in team_data[year]:
        for typ in ['all', 'bowl']:
            
            season_stats = team_data[year][team][typ]
            season_stats['Takeaways'] = season_stats['PassIntDef'] \
                    + season_stats['FumblesDef']
            season_stats['Giveaways'] = season_stats['PassIntOff'] \
                    + season_stats['FumblesOff']
            season_stats['TOMargin'] = season_stats['Takeaways'] \
                    - season_stats['Giveaways']
'''
# print team stats for a year
for stat in team_data[2013]['Michigan']['all'].items():
    print stat
'''
# recast data in terms of underdogs and favorites
for year in range(2002,2014):
    N = len(raw['bowls'][year]['Home Team'])
    
    for team_status in ['Underdog', 'Favored']:
        if team_status not in raw['bowls'][year]:
            raw['bowls'][year][team_status] = list()
            raw['bowls'][year][team_status + ' Score'] = list()

    # go through all games
    for i in range(N):
        
        # extract the line
        line = raw['bowls'][year]['Line'][i]

        # determine the index of the favorite
        if line > 0:
            index = 0
        elif line < 0:
            index = 1
        else:
            # flip a coin
            if np.random.random() > 0.5:
                index = 0
            else:
                index = 1

        # extract data
        home = raw['bowls'][year]['Home Team'][i]
        home_score = raw['bowls'][year]['Home Score'][i]
        visit = raw['bowls'][year]['Visitor'][i]
        visit_score = raw['bowls'][year]['Visitor Score'][i]
        
        # create lists for iterationd
        teams = [home, visit]
        scores = [home_score, visit_score]
        favor = ['Favored', 'Underdog']

        # add data to dictionary
        for j in range(2):
            idx = abs(index - j)
            raw['bowls'][year][ favor[j] ].append(teams[idx])
            raw['bowls'][year][favor[j]+' Score'].append(scores[idx])
    

# form X-data
X = list()
Y = list()
for year in range(2002,2014):
    N = len(raw['bowls'][year]['Home Team'])
    for i in range(N):
        xvector = list()
        xheader = list()
        for team_status in ['Home Team', 'Visitor']:
            
            # get team name
            team = raw['bowls'][year][team_status][i]
            if team not in team_data[year]:
                team = alt_name[team]

            # add stats
            for typ in ['all', 'bowl']:
                for stat in team_data[year][team][typ].items():
                    
                    # determine name for stat
                    stat_name = team_status + ' ' + typ + stat[0]
                    xheader.append(stat_name)

                    # add stat value
                    xvector.append(stat[1])

        # get teams involved in matchup
        bowl_year = raw['bowls'][year]
        teams = [bowl_year['Home Team'][i], bowl_year['Visitor'][i]]
        for k, team in enumerate(teams):
            if team not in team_data[year]:
                teams[k] = alt_name[team]
        
        # get matchup statistics
        for k in [0, 1]:
            offense = teams[k]
            defense = teams[1-k]
            for typ in ['all', 'bowl']:
                for attack in ['Pass', 'Rush']:
                    
                    # determine name for stat
                    stat_name = ''
                    if k == 0:
                        stat_name += 'Home Team'
                    else:
                        stat_name += 'Visitor'
                    stat_name += ' Matchup ' + attack + ' ' + typ
                    xheader.append(stat_name)

                    # get stat value
                    off_stats = team_data[year][offense][typ]
                    def_stats = team_data[year][defense][typ]
                    val = off_stats[attack + 'YdsOff'] \
                            + def_stats[attack + 'YdsDef']
                    xvector.append(val)

        # add line
        line = raw['bowls'][year]['Line'][i]
        xheader.append('Line')
        xvector.append(line)

        # add result
        home_score = raw['bowls'][year]['Home Score'][i]
        visit_score = raw['bowls'][year]['Visitor Score'][i]
        if home_score > visit_score:
            Y.append(1)
        else:
            Y.append(0)

        # add feature vector
        X.append(xvector)

N = len(Y)
with open('formed_data.csv', 'w') as fh:
    for header in xheader:
        fh.write(header + ',')
    fh.write('Result\n')
    for i in range(N):
        for j in range(len(X[i])):
            fh.write(str(X[i][j]) + ',')
        
        fh.write(str(Y[i]))
        if i != N-1:
            fh.write('\n')
